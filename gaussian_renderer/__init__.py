#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path
import time

import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from utils.entropy_models import Low_bound


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_transmit_mask()[visible_mask]  # [N_vis, 10, 1]

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None

    if is_training:
        # Apply SOA enhancement if enabled and past initial training stage
        if pc.use_soa and step > 3000:
            # Calculate feature enhancements using SOA
            augmented_features = []
            for i in range(pc.soa_m):
                eigen_projection = torch.matmul(feat, pc.soa_top_eigenvectors[:,i:i+1])  # [N, 1]
                combined = torch.cat([feat, eigen_projection.expand(-1, pc.feat_dim)], dim=-1)  # [N, 2D]
                enhanced_feat = pc.soa_mlp[i](combined)  # [N, D]
                augmented_features.append(enhanced_feat)
            
            # Store original features for quantization
            orig_feat = feat.clone()
            
            # Use enhanced features for rendering, but quantize original ones
            feat = feat + sum(augmented_features) / pc.soa_m
        
        if step > 3000 and step <= 10000:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * 1
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * 0.001
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * 0.2

        if step == 10000:
            pc.update_anchor_bound()
            
            # First update of SOA components
            if pc.use_soa:
                with torch.no_grad():
                    pc.update_second_order_components(pc._anchor_feat)

        if step > 10000:
            # for rendering
            feat_context_orig = pc.calc_interp_feat(anchor)
            feat_context_ttl = pc.get_grid_mlp.forward(feat_context_orig, cli=pc.current_lmbda_idx)
            ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

            Q_feat_adj, Q_scaling_adj, Q_offsets_adj = Q_base[0], Q_base[1], Q_base[2]
            Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
            Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

            # Adjust quantization based on level - sharper quantization for higher levels
            Q_feat = Q_feat_basic / (3 ** pc.current_lmbda_idx)
            Q_scaling = Q_scaling_basic / (3 ** pc.current_lmbda_idx)
            Q_offsets = 0.2 * (1 + torch.tanh(Q_offsets_adj))

            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.view(-1, pc.n_offsets, 3)

            # for entropy
            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
            anchor_chosen = pc.get_anchor[choose_idx]
            feat_chosen_orig = pc._anchor_feat[choose_idx]
            grid_offsets_chosen_orig = pc._offset[choose_idx]
            grid_scaling_chosen_orig = pc.get_scaling[choose_idx]
            binary_grid_masks_chosen = pc.get_transmit_mask()[choose_idx]  # [N_vis, 10, 1]
            mask_anchor_chosen = pc.get_mask_anchor()[choose_idx]  # [N_vis, 1]

            feat_context_orig = pc.calc_interp_feat(anchor_chosen)
            feat_context_ttl = pc.get_grid_mlp.forward(feat_context_orig, cli=pc.current_lmbda_idx)
            ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, _, _, _ = \
                torch.split(ctx_info_old, split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            mean_offsets, scale_offsets = feat_context_ttl[3]
            Q_feat_adj, Q_scaling_adj, Q_offsets_adj = Q_base[0], Q_base[1], Q_base[2]
            Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
            Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

            Q_feat = Q_feat_basic / (3 ** pc.current_lmbda_idx)
            Q_scaling = Q_scaling_basic / (3 ** pc.current_lmbda_idx)
            Q_offsets = 0.2 * (1 + torch.tanh(Q_offsets_adj.contiguous().view(-1, pc.n_offsets, 3)))

            feat_chosen_q = feat_chosen_orig + torch.empty_like(feat_chosen_orig).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling_chosen_q = grid_scaling_chosen_orig + torch.empty_like(grid_scaling_chosen_orig).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets_chosen_q = (grid_offsets_chosen_orig + torch.empty_like(grid_offsets_chosen_orig).uniform_(-0.5, 0.5) * Q_offsets).view(-1, 3 * pc.n_offsets)
            mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen_q, torch.cat([mean, scale, prob], dim=-1))
            if 1:
                probs = torch.stack([prob, prob_adj], dim=-1)
                probs = torch.softmax(probs, dim=-1)

                binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).view(-1, 3*pc.n_offsets)

                bit_feat = pc.EG_mix_prob_2.forward(feat_chosen_q,
                                                    mean, mean_adj,
                                                    scale, scale_adj,
                                                    probs[..., 0], probs[..., 1],
                                                    Q=Q_feat, x_mean=pc._anchor_feat.mean())
                bit_feat = bit_feat * mask_anchor_chosen
                bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen_q, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
                bit_scaling = bit_scaling * mask_anchor_chosen
                bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen_q, mean_offsets, scale_offsets, Q_offsets.view(-1, 3*pc.n_offsets), pc._offset.mean())
                bit_offsets = bit_offsets * binary_grid_masks_chosen

            if pc.current_lmbda_idx > 0:  # pc.current_lmbda_idx > 0:

                prob_feat = pc.get_grid_mlp.forward_prob_feat(feat_context_orig, feat_chosen_orig, Q_feat, pc.current_lmbda_idx, pc._anchor_feat.mean())  # [N, 50]
                bit_feat_inc = -torch.log2(Low_bound.apply(prob_feat)) * mask_anchor_chosen

                prob_scaling = pc.get_grid_mlp.forward_prob_scaling(feat_context_orig, grid_scaling_chosen_orig, Q_scaling, pc.current_lmbda_idx, pc.get_scaling.mean())  # [N, 50]
                bit_scaling_inc = -torch.log2(Low_bound.apply(prob_scaling)) * mask_anchor_chosen

                # binary_grid_masks_chosen: [N_chosen, 30]
                mask_mode_anchor = mask_anchor_chosen - pc.get_mask_anchor(cli=pc.current_lmbda_idx-1)[choose_idx].detach()  # 如果为0，说明前一个λ有参考。  # [N_chosen, 1]
                mask_mode_Gaussian = binary_grid_masks_chosen - (pc.get_transmit_mask(cli=pc.current_lmbda_idx-1)[choose_idx]).repeat(1, 1, 3).view(-1, 3*pc.n_offsets).detach()  # 如果为0，说明前一个λ有参考。  # [N_chosen, 30]
                assert torch.min(mask_mode_anchor).item() >= 0 and torch.max(mask_mode_anchor).item() <= 1
                assert torch.min(mask_mode_Gaussian).item() >= 0 and torch.max(mask_mode_Gaussian).item() <= 1

                bit_feat = bit_feat * (mask_mode_anchor) + bit_feat_inc * (1 - mask_mode_anchor)
                bit_scaling = bit_scaling * (mask_mode_anchor) + bit_scaling_inc * (1 - mask_mode_anchor)
                bit_offsets = bit_offsets * (mask_mode_Gaussian)

            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()

        feat_context_orig = pc.calc_interp_feat(anchor)
        feat_context_ttl = pc.get_grid_mlp.forward(feat_context_orig, cli=pc.current_lmbda_idx)
        ctx_info_old, Q_base = feat_context_ttl[0], feat_context_ttl[1]

        Q_feat_adj, Q_scaling_adj, Q_offsets_adj = Q_base[0], Q_base[1], Q_base[2]
        Q_feat_basic = 1 * (1 + torch.tanh(Q_feat_adj.contiguous()))
        Q_scaling_basic = 0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()))

        Q_offsets = 0.2 * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)  # [N_visible_anchor, 10, 3]
        Q_feat = Q_feat_basic / (3 ** pc.current_lmbda_idx)
        Q_scaling = Q_scaling_basic / (3 ** pc.current_lmbda_idx)

        feat = STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean()).detach()
        grid_scaling = STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean()).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()

        torch.cuda.synchronize(); time_sub = time.time() - t1
    else:
        # If using SOA in decoded version
        if pc.use_soa:
            # Calculate feature enhancements using SOA
            augmented_features = []
            for i in range(pc.soa_m):
                eigen_projection = torch.matmul(feat, pc.soa_top_eigenvectors[:,i:i+1])  # [N, 1]
                combined = torch.cat([feat, eigen_projection.expand(-1, pc.feat_dim)], dim=-1)  # [N, 2D]
                enhanced_feat = pc.soa_mlp[i](combined)  # [N, D]
                augmented_features.append(enhanced_feat)
            
            # Use enhanced features for rendering
            feat = feat + sum(augmented_features) / pc.soa_m

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    binary_grid_masks_pergaussian = binary_grid_masks.view(-1, 1)
    if is_training:
        opacity = opacity * binary_grid_masks_pergaussian[mask]
        scaling = scaling * binary_grid_masks_pergaussian[mask]
    else:
        the_mask = (binary_grid_masks_pergaussian[mask]).to(torch.bool)
        the_mask = the_mask[:, 0]
        xyz = xyz[the_mask]
        color = color[the_mask]
        opacity = opacity[the_mask]
        scaling = scaling[the_mask]
        rot = rot[the_mask]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0

import os

iterations = 30_000  # or 40_000, or at your will
for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
    one_cmd = (f'CUDA_VISIBLE_DEVICES={0} python train.py '
               f'-s data/bungeenerf/{scene} '
               f'--eval '
               f'--lod 30 '
               f'--voxel_size 0 '
               f'--update_init_factor 128 '
               f'-m output/bungeenerf/{scene} '
               f'--lmbda_list 8 4 0.5 '
               f'--iterations {iterations} '
               f'--position_lr_max_steps {iterations} '
               f'--offset_lr_max_steps {iterations} '
               f'--mask_lr_max_steps {iterations} '
               f'--mlp_opacity_lr_max_steps {iterations} '
               f'--mlp_cov_lr_max_steps {iterations} '
               f'--mlp_color_lr_max_steps {iterations} '
               f'--mlp_featurebank_lr_max_steps {iterations} '
               f'--encoding_xyz_lr_max_steps {iterations} '
               f'--mlp_grid_lr_max_steps {iterations} '
               f'--mlp_deform_lr_max_steps {iterations} '
               f'--mask_lr_final {0.00007}')
    os.system(one_cmd)

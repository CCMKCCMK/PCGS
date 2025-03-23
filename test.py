import os
import subprocess

iterations = 30_000  # or 40_000, or at your will
for scene in ['truck', 'train']:
    one_cmd = (f'CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(\'CUDA Available:\', torch.cuda.is_available())"')
    os.system(one_cmd)
    
    # Apply a fix for the prod operation issue
    one_cmd = (f'CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=0 python -m torch.utils.collect_env')
    os.system(one_cmd)
    
    one_cmd = (f'CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=0 python train.py '
               f'-s data/tandt/{scene} '
               f'--eval '
               f'--lod 0 '
               f'--voxel_size 0.01 '
               f'--update_init_factor 16 '
               f'--fix_scaling_bug '  # Add this flag for our fix
               f'-m output/tandt/{scene} '
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
               f'--mask_lr_final {0.0001}')
    os.system(one_cmd)
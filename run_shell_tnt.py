import os
from datetime import datetime

# Set environment variable to force compatible CUDA architecture
# This is a workaround for newer GPUs (Compute Capability 8.9) with older PyTorch
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'

# Random seed for reproducibility
RANDOM_SEED = 42

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    # for cuda, scene in enumerate(['truck', 'train']):
    for cuda, scene in enumerate(['truck']):
        mask_lr_final = 0.0001 * lmbda / 0.001
        # Use absolute path to avoid path resolution issues
        data_path = f'/workspace/HAC/data/tandt/{scene}'

        # Hierarchical parameters: carefully tuned to avoid GPU memory overflow
        # Level 1 voxel scale 8.0 -> reduces Level 1 anchors by ~512x (8^3)
        # Level 2 per Level 1 = 8 -> generates only 8 Level 2 anchors per Level 1
        # Total Level 2 = ~49K × 8 = ~392K anchors (manageable)
        # Compression: storing only 1/512 of original anchors as Level 1
        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{lmbda}_2level_test_1-4_no3/ --lmbda {lmbda} --mask_lr_final {mask_lr_final} --use_hierarchical --intra_anchor_type mlp --level1_voxel_scale 1.0 --level2_per_level1 1'

        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{lmbda}_2level_moment_test_1/ --lmbda {lmbda} --mask_lr_final {mask_lr_final} --use_hierarchical --intra_anchor_type moment_matching --level1_voxel_scale 1.0'


        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py --use_gated_mlp -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{lmbda}_base_gate-10-MLP --lmbda {lmbda} --mask_lr_final {mask_lr_final}'

        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py  -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{lmbda}_Q=0.1 --lmbda {lmbda} --mask_lr_final {mask_lr_final}'

        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py  --use_spatial_context -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{lmbda}_Q=0.1_Context_fix --lmbda {lmbda} --mask_lr_final {mask_lr_final}'

        now = datetime.now()
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py  -s {data_path} --eval  --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/_Q=0.1_afterTanh_PointNet{now:%Y-%m-%d}_\${now:%H-%M-%S}_{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final} --seed {RANDOM_SEED}'
        # one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py  -s {data_path} --eval --resolution 320 --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_test/{scene}/{now:%Y-%m-%d}_\${now:%H-%M-%S}_{lmbda}_Q=0.1_attention --lmbda {lmbda} --mask_lr_final {mask_lr_final}'



        print(f"Running: {one_cmd}")
        os.system(one_cmd)

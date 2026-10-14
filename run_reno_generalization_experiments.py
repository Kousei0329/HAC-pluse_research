import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# RENO's torchsparse kernels JIT-compile via NVRTC at runtime; this sandbox's own PyTorch install
# only bundles an 11.6 NVRTC (no sm_89/Ada support), so it fails invalid --gpu-architecture unless
# a newer NVRTC (here, HAC_mamba_env's) is put ahead of it in the loader search path.
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# Follow-up to the earlier high/low-rate-only RENO-on run: fills in the 3 middle lambda points
# (0.001 and 0.005 already covered) across the same tandt -> db -> nerf_synthetic order, same
# causal_knn_K=8 + MLP-fp16 + RENO-on config.
LAMBDAS = [0.002, 0.003, 0.004]
CAUSAL_KNN_K = 8
GPU_ID = 0
EXPERIMENT_NAME = "generalization_K8_fp16_reno_midlambda"
RANDOM_SEED = 42

DATASET_CONFIGS = {
    'tandt': {
        'scenes': ['truck', 'train'],
        'data_base': '/workspace/HAC/data/tandt',
        'voxel_size': 0.01,
        'mask_lr_final_base': 0.0001,
        'mask_lr_final_max': None,
        'update_init_factor': 16,
        'lod': 0,
    },
    'db': {
        'scenes': ['drjohnson', 'playroom'],
        'data_base': '/workspace/HAC/data/db',
        'voxel_size': 0.005,
        'mask_lr_final_base': 0.00008,
        'mask_lr_final_max': None,
        'update_init_factor': 16,
        'lod': 0,
    },
    'nerf_synthetic': {
        'scenes': ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'],
        'data_base': '/workspace/HAC/data/nerf_synthetic',
        'voxel_size': 0.001,
        'mask_lr_final_base': 0.00008,
        'mask_lr_final_max': None,
        'update_init_factor': 4,
        'lod': 0,
    },
    # mipnerf360/bungeenerf excluded, same as run_generalization_experiments.py: their large
    # outdoor/city scenes OOM'd on this 16GB GPU even without RENO, and RENO only adds to memory
    # use, so they were not retried here.
}

ACTIVE_DATASETS = ['tandt', 'db', 'nerf_synthetic']

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def resolve_active_scenes():
    pairs = []
    for dataset_name in ACTIVE_DATASETS:
        for scene in DATASET_CONFIGS[dataset_name]['scenes']:
            pairs.append((dataset_name, scene))
    return pairs


def create_command(lmbda, scene, dataset_name, gpu_id, seed):
    cfg = DATASET_CONFIGS[dataset_name]
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001
    if cfg['mask_lr_final_max'] is not None:
        mask_lr_final = min(mask_lr_final, cfg['mask_lr_final_max'])

    data_path = f"{cfg['data_base']}/{scene}"
    output_path = f"{EXPERIMENT_DIR}/{dataset_name}/{scene}/lambda_{lmbda}"

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'-s {data_path} '
        f'--eval '
        f'--lod {cfg["lod"]} '
        f'--voxel_size {cfg["voxel_size"]} '
        f'--update_init_factor {cfg["update_init_factor"]} '
        f'--iterations 30_000 '
        f'-m {output_path} '
        f'--lmbda {lmbda} '
        f'--mask_lr_final {mask_lr_final} '
        f'--seed {seed} '
        f'--causal_knn_K {CAUSAL_KNN_K} '
        f'--quantize_mlp_fp16 '
        # RENO stays on its default (True) here -- this is the same causal_knn_K=8 + fp16 config,
        # with RENO's neural anchor-position codec enabled instead of the --no_use_reno ablation.
    )
    return cmd, output_path


def run_experiment(lmbda, scene, dataset_name, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, scene, dataset_name, gpu_id, seed)

    print(f"\n{'='*80}")
    print(f"Starting experiment {exp_num}/{total}:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Scene:   {scene}")
    print(f"  Lambda:  {lmbda}")
    print(f"  GPU:     {gpu_id}")
    print(f"  Output:  {output_path}")
    print(f"{'='*80}\n")

    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)

    start_time = time.time()
    print(f"Running command:\n{cmd}\n")

    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Completed experiment {exp_num}/{total}:")
    print(f"  Dataset: {dataset_name}, Scene: {scene}, Lambda: {lmbda}")
    print(f"  Time: {elapsed/60:.2f} minutes")
    print(f"  Return code: {result.returncode}")
    print(f"  Log: {log_file}")
    print(f"{'='*80}\n")

    return {
        'dataset': dataset_name,
        'scene': scene,
        'lmbda': lmbda,
        'output_path': output_path,
        'log_file': log_file,
        'duration_minutes': elapsed / 60,
        'return_code': result.returncode,
    }


def main():
    active_scenes = resolve_active_scenes()

    experiments = [
        {'dataset': ds, 'scene': sc, 'lmbda': lmbda}
        for ds, sc in active_scenes
        for lmbda in LAMBDAS
    ]
    total = len(experiments)

    config = {
        "timestamp": TIMESTAMP,
        "lambdas": LAMBDAS,
        "causal_knn_K": CAUSAL_KNN_K,
        "quantize_mlp_fp16": True,
        "use_reno": True,
        "active_scenes": [{"dataset": ds, "scene": sc} for ds, sc in active_scenes],
        "gpu_id": GPU_ID,
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Generalization Suite (RENO on, mid-lambda fill-in): causal_knn_K=8 + fp16 MLP + RENO")
    print(f"{'='*80}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Active scenes (in order):")
    for ds, sc in active_scenes:
        print(f"  [{ds}] {sc}")
    print(f"GPU: {GPU_ID}")
    print(f"Total experiments: {total}")
    print(f"{'='*80}\n")

    completed = []
    for i, exp in enumerate(experiments, 1):
        result = run_experiment(
            exp['lmbda'], exp['scene'], exp['dataset'],
            GPU_ID, RANDOM_SEED, i, total
        )
        completed.append(result)

    print(f"\n{'='*80}")
    print(f"All {len(completed)} experiments completed!")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print(f"{'='*80}\n")

    summary = {
        'experiment_dir': EXPERIMENT_DIR,
        'config': config,
        'completed_experiments': completed,
        'total_duration_minutes': sum(r['duration_minutes'] for r in completed),
    }
    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results summary: {EXPERIMENT_DIR}/results_summary.json")
    print(f"Total execution time: {summary['total_duration_minutes']:.2f} minutes\n")


if __name__ == "__main__":
    main()

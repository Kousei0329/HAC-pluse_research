import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# --- Global Settings ---
LAMBDAS = [0.001, 0.002, 0.003, 0.004, 0.005]
CAUSAL_KNN_K_VALUES = [4, 8, 16, 32]
GPU_ID = 0
EXPERIMENT_NAME = "causal_knn_K_comparison"
RANDOM_SEED = 42

DATASET_CONFIG = {
    'scene': 'truck',
    'data_base': '/workspace/HAC/data/tandt',
    'voxel_size': 0.01,
    'mask_lr_final_base': 0.0001,
    'update_init_factor': 16,
    'lod': 0,
}

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def create_command(lmbda, K, gpu_id, seed):
    cfg = DATASET_CONFIG
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001

    data_path = f"{cfg['data_base']}/{cfg['scene']}"
    output_path = f"{EXPERIMENT_DIR}/K_{K}/lambda_{lmbda}"

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
        f'--causal_knn_K {K} '
        f'--no_use_reno '
        # use_causal_knn / use_anchor_cond_norm stay at their argparse defaults (True). No
        # pruning/quantization/3GMM here -- this sweep is isolating causal_knn_K alone.
    )
    return cmd, output_path


def run_experiment(lmbda, K, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, K, gpu_id, seed)

    print(f"\n{'='*80}")
    print(f"Starting experiment {exp_num}/{total}:")
    print(f"  Scene:   truck")
    print(f"  K:       {K}")
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
    print(f"  K: {K}, Lambda: {lmbda}")
    print(f"  Time: {elapsed/60:.2f} minutes")
    print(f"  Return code: {result.returncode}")
    print(f"  Log: {log_file}")
    print(f"{'='*80}\n")

    return {
        'K': K,
        'lmbda': lmbda,
        'output_path': output_path,
        'log_file': log_file,
        'duration_minutes': elapsed / 60,
        'return_code': result.returncode,
    }


def main():
    experiments = [
        {'K': K, 'lmbda': lmbda}
        for K in CAUSAL_KNN_K_VALUES
        for lmbda in LAMBDAS
    ]
    total = len(experiments)

    config = {
        "timestamp": TIMESTAMP,
        "lambdas": LAMBDAS,
        "causal_knn_K_values": CAUSAL_KNN_K_VALUES,
        "scene": DATASET_CONFIG['scene'],
        "gpu_id": GPU_ID,
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "use_reno": False,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Causal KNN K Comparison Suite (Sequential Execution)")
    print(f"{'='*80}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"K values: {CAUSAL_KNN_K_VALUES}")
    print(f"Scene: truck (RENO disabled)")
    print(f"GPU: {GPU_ID}")
    print(f"Total experiments: {total}")
    print(f"{'='*80}\n")

    completed = []
    for i, exp in enumerate(experiments, 1):
        result = run_experiment(exp['lmbda'], exp['K'], GPU_ID, RANDOM_SEED, i, total)
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

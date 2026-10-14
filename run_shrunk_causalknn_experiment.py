import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# causal_knn + fp8 (get_mlp_size now counts causal_knn) came out ~92-224% worse than baseline in
# BD-Rate because causal_knn's correction network (hidden_mult=8 default) alone has 484K params --
# more than baseline's entire MLP budget. This shrinks causal_knn's hidden width (causal_knn_hidden_mult)
# and mlp_grid's hidden width (mlp_grid_hidden_mult) to trade capacity for size, targeting a total
# MLPs size back near baseline's ~0.20MB (nerf_synthetic) while keeping causal_knn ENABLED.
#   causal_knn_hidden_mult=2 -> causal_knn: 484129 -> 65953 params (7.3x smaller)
#   mlp_grid_hidden_mult=4   -> mlp_grid:   244825 ->  82525 params (3.0x smaller)
# Combined + mlp_deform(9990) + fp8(8 bits/param) + fixed heads(fp32) -> ~0.235MB, close to
# baseline's 0.2028MB for nerf_synthetic.
LAMBDAS = [0.001, 0.005]
CAUSAL_KNN_K = 8
CAUSAL_KNN_HIDDEN_MULT = 2
MLP_GRID_HIDDEN_MULT = 4
GPU_ID = 0
EXPERIMENT_NAME = "shrunk_causalknn_fp8"
RANDOM_SEED = 42

DATASET_CONFIG = {
    'scene': 'hotdog',
    'data_base': '/workspace/HAC/data/nerf_synthetic',
    'voxel_size': 0.001,
    'mask_lr_final_base': 0.00008,
    'update_init_factor': 4,
    'lod': 0,
}

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR = f"outputs/experiments/{EXPERIMENT_NAME}_{TIMESTAMP}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def create_command(lmbda, gpu_id, seed):
    cfg = DATASET_CONFIG
    mask_lr_final = cfg['mask_lr_final_base'] * lmbda / 0.001
    data_path = f"{cfg['data_base']}/{cfg['scene']}"
    output_path = f"{EXPERIMENT_DIR}/lambda_{lmbda}"

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} /opt/conda/envs/HAC_plux_env/bin/python train.py '
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
        f'--causal_knn_hidden_mult {CAUSAL_KNN_HIDDEN_MULT} '
        f'--mlp_grid_hidden_mult {MLP_GRID_HIDDEN_MULT} '
        f'--quantize_mlp_fp8 '
        # RENO stays on its default (True).
    )
    return cmd, output_path


def run_experiment(lmbda, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, gpu_id, seed)
    print(f"\n{'='*80}\nStarting experiment {exp_num}/{total}: hotdog lambda={lmbda}\n{'='*80}\n")
    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    print(f"Running command:\n{cmd}\n")
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time
    print(f"\n{'='*80}\nCompleted experiment {exp_num}/{total}: lambda={lmbda}  "
          f"Time: {elapsed/60:.2f}min  Return code: {result.returncode}\n{'='*80}\n")
    return {'lmbda': lmbda, 'output_path': output_path, 'log_file': log_file,
            'duration_minutes': elapsed / 60, 'return_code': result.returncode}


def main():
    total = len(LAMBDAS)
    config = {
        "timestamp": TIMESTAMP, "lambdas": LAMBDAS, "causal_knn_K": CAUSAL_KNN_K,
        "causal_knn_hidden_mult": CAUSAL_KNN_HIDDEN_MULT, "mlp_grid_hidden_mult": MLP_GRID_HIDDEN_MULT,
        "quantize_mlp_fp8": True, "use_reno": True, "scene": DATASET_CONFIG['scene'],
        "gpu_id": GPU_ID, "experiment_name": EXPERIMENT_NAME, "random_seed": RANDOM_SEED,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}\nShrunk causal_knn + shrunk mlp_grid + fp8 on hotdog\n{'='*80}")
    print(f"causal_knn_hidden_mult={CAUSAL_KNN_HIDDEN_MULT}, mlp_grid_hidden_mult={MLP_GRID_HIDDEN_MULT}")
    print(f"Lambdas: {LAMBDAS}\n{'='*80}\n")

    completed = []
    for i, lmbda in enumerate(LAMBDAS, 1):
        completed.append(run_experiment(lmbda, GPU_ID, RANDOM_SEED, i, total))

    print(f"\n{'='*80}\nAll {len(completed)} experiments completed!\n{'='*80}\n")
    summary = {'experiment_dir': EXPERIMENT_DIR, 'config': config, 'completed_experiments': completed,
               'total_duration_minutes': sum(r['duration_minutes'] for r in completed)}
    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results summary: {EXPERIMENT_DIR}/results_summary.json")


if __name__ == "__main__":
    main()

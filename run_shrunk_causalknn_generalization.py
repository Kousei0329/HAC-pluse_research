import os
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# Follow-up to run_shrunk_causalknn_experiment.py (hotdog only): that config -- causal_knn ON with
# hidden_mult=2 (484K->66K params), mlp_grid hidden_mult=4 (245K->83K params), fp8 quantization --
# beat baseline on hotdog (BD-Rate -3.8%, vs +91.6% with the unshrunk architecture). This checks
# whether that generalizes across the other 11 scenes (hotdog itself already done, not rerun here).
LAMBDAS = [0.001, 0.005]
CAUSAL_KNN_K = 8
CAUSAL_KNN_HIDDEN_MULT = 2
MLP_GRID_HIDDEN_MULT = 4
GPU_ID = 0
EXPERIMENT_NAME = "shrunk_causalknn_fp8_generalization"
RANDOM_SEED = 42

DATASET_CONFIGS = {
    'tandt': {
        'scenes': ['truck', 'train'],
        'data_base': '/workspace/HAC/data/tandt',
        'voxel_size': 0.01,
        'mask_lr_final_base': 0.0001,
        'update_init_factor': 16,
        'lod': 0,
    },
    'db': {
        'scenes': ['drjohnson', 'playroom'],
        'data_base': '/workspace/HAC/data/db',
        'voxel_size': 0.005,
        'mask_lr_final_base': 0.00008,
        'update_init_factor': 16,
        'lod': 0,
    },
    'nerf_synthetic': {
        # hotdog excluded -- already run separately (see run_shrunk_causalknn_experiment.py)
        'scenes': ['chair', 'drums', 'ficus', 'lego', 'materials', 'mic', 'ship'],
        'data_base': '/workspace/HAC/data/nerf_synthetic',
        'voxel_size': 0.001,
        'mask_lr_final_base': 0.00008,
        'update_init_factor': 4,
        'lod': 0,
    },
}

ACTIVE_DATASETS = ['nerf_synthetic', 'tandt', 'db']

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
    data_path = f"{cfg['data_base']}/{scene}"
    output_path = f"{EXPERIMENT_DIR}/{dataset_name}/{scene}/lambda_{lmbda}"

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


def run_experiment(lmbda, scene, dataset_name, gpu_id, seed, exp_num, total):
    cmd, output_path = create_command(lmbda, scene, dataset_name, gpu_id, seed)
    print(f"\n{'='*80}\nStarting experiment {exp_num}/{total}: [{dataset_name}] {scene} lambda={lmbda}\n{'='*80}\n")
    log_file = f"{output_path}/training.log"
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    print(f"Running command:\n{cmd}\n")
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time
    print(f"\n{'='*80}\nCompleted experiment {exp_num}/{total}: [{dataset_name}] {scene} lambda={lmbda}  "
          f"Time: {elapsed/60:.2f}min  Return code: {result.returncode}\n{'='*80}\n")
    return {'dataset': dataset_name, 'scene': scene, 'lmbda': lmbda, 'output_path': output_path,
            'log_file': log_file, 'duration_minutes': elapsed / 60, 'return_code': result.returncode}


def main():
    active_scenes = resolve_active_scenes()
    experiments = [{'dataset': ds, 'scene': sc, 'lmbda': lmbda}
                    for ds, sc in active_scenes for lmbda in LAMBDAS]
    total = len(experiments)

    config = {
        "timestamp": TIMESTAMP, "lambdas": LAMBDAS, "causal_knn_K": CAUSAL_KNN_K,
        "causal_knn_hidden_mult": CAUSAL_KNN_HIDDEN_MULT, "mlp_grid_hidden_mult": MLP_GRID_HIDDEN_MULT,
        "quantize_mlp_fp8": True, "use_reno": True,
        "active_scenes": [{"dataset": ds, "scene": sc} for ds, sc in active_scenes],
        "gpu_id": GPU_ID, "experiment_name": EXPERIMENT_NAME, "random_seed": RANDOM_SEED,
    }
    with open(f"{EXPERIMENT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}\nShrunk causal_knn + shrunk mlp_grid + fp8: generalization sweep\n{'='*80}")
    print(f"causal_knn_hidden_mult={CAUSAL_KNN_HIDDEN_MULT}, mlp_grid_hidden_mult={MLP_GRID_HIDDEN_MULT}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Active scenes (in order): {active_scenes}")
    print(f"Total experiments: {total}\n{'='*80}\n")

    completed = []
    for i, exp in enumerate(experiments, 1):
        completed.append(run_experiment(exp['lmbda'], exp['scene'], exp['dataset'], GPU_ID, RANDOM_SEED, i, total))

    print(f"\n{'='*80}\nAll {len(completed)} experiments completed!\n{'='*80}\n")
    summary = {'experiment_dir': EXPERIMENT_DIR, 'config': config, 'completed_experiments': completed,
               'total_duration_minutes': sum(r['duration_minutes'] for r in completed)}
    with open(f"{EXPERIMENT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results summary: {EXPERIMENT_DIR}/results_summary.json")


if __name__ == "__main__":
    main()

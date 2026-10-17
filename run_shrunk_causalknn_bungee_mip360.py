import os
import shutil
import subprocess
from datetime import datetime
import time
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/HAC_mamba_env/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# --- Global Settings ---
# Extends the shrunk-causal_knn + shrunk-mlp_grid quantization-width ablation (see
# run_shrunk_causalknn_generalization.py: tandt/db/nerf_synthetic already done) to BungeeNeRF and
# Mip-NeRF360. Deliberately trains with NO quantize_mlp_* flag -- those round-trip the MLP
# weights in-place before scene.save(), permanently destroying the fp32 checkpoint. Quantization
# variants (fp32/int8/...) are evaluated post-hoc from this clean checkpoint via
# eval_quant_variant.py (see run_eval_quant_bungee_mip360.py), matching the workflow this run's
# own repo history switched to after getting burned by baked-in fp8 checkpoints.
LAMBDAS = [float(x) for x in os.environ.get('LAMBDAS', '0.001,0.005').split(',')]
CAUSAL_KNN_K = 8
CAUSAL_KNN_HIDDEN_MULT = 2
MLP_GRID_HIDDEN_MULT = 4
GPU_ID = int(os.environ.get('GPU_ID', 0))
EXPERIMENT_NAME = "shrunk_causalknn_bungee_mip360"
RANDOM_SEED = 42
# HAC_plux_env (used by the other shrunk-arch scripts) OOMs immediately (33GB single alloc) on
# bungeenerf's --voxel_size 0 auto-compute path -- none of the other shrunk-arch runs (tandt/db/
# nerf_synthetic) ever exercise that path, they all pass an explicit nonzero voxel_size. This
# repo's own earlier RD-sweep runs proved HAC_env_pytorch2 handles bungeenerf's voxel_size=0 +
# lod=30 combination correctly, so use that env instead for this script.
PYTHON_BIN = "/opt/conda/envs/HAC_env_pytorch2/bin/python"

# flowers/treehill excluded per instructions (separate-license download); data paths are this
# repo's own local data/ (relative), not the /workspace/HAC/data/ convention the other
# shrunk-arch scripts were written against on a different machine.
DATASET_CONFIGS = {
    'bungeenerf': {
        'scenes': ['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome'],
        'data_base': 'data/bungeenerf',
        'voxel_size': 0,
        'mask_lr_final_base': 0.0001,
        'mask_lr_final_max': None,
        'update_init_factor': 128,
        'lod': 30,
    },
    'mipnerf360': {
        'scenes': ['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai'],
        'data_base': 'data/mipnerf360',
        'voxel_size': 0.001,
        'mask_lr_final_base': 0.0005,
        'mask_lr_final_max': 0.0015,
        'update_init_factor': 16,
        'lod': 0,
    },
}
ACTIVE_DATASETS = ['bungeenerf', 'mipnerf360']

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Pinned (not re-timestamped) so parallel GPU0/GPU1 invocations -- split by LAMBDAS env var --
# write into the same directory and their skip-if-done checks see each other's progress. Matches
# the dir the single-GPU run already created (outputs/experiments/shrunk_causalknn_bungee_mip360_
# 2026-10-15_07-09-07), where amsterdam/lambda_0.001 was left mid-training (no point_cloud yet)
# when that run was stopped to switch to this parallel split -- it will be retried cleanly.
EXPERIMENT_DIR = os.environ.get(
    'EXPERIMENT_DIR',
    "outputs/experiments/shrunk_causalknn_bungee_mip360_2026-10-15_07-09-07",
)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

MIN_FREE_GB_SOFT = 15
MIN_FREE_GB_HARD = 6


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
        f'CUDA_VISIBLE_DEVICES={gpu_id} {PYTHON_BIN} train.py '
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
        # No --quantize_mlp_* flag: keep the saved checkpoint clean fp32.
        # RENO stays on its default (True).
    )
    return cmd, output_path


def free_gb():
    return shutil.disk_usage('.').free / (1024 ** 3)


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

    # Merge with any existing config.json's lambdas -- two parallel invocations (split by the
    # LAMBDAS env var, one per GPU) each only know their own slice and would otherwise clobber
    # each other's entry with a last-write-wins race.
    existing_lambdas = []
    config_path = f"{EXPERIMENT_DIR}/config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing_lambdas = json.load(f).get("lambdas", [])
        except (json.JSONDecodeError, OSError):
            pass
    all_lambdas = sorted(set(existing_lambdas) | set(LAMBDAS))

    config = {
        "timestamp": TIMESTAMP, "lambdas": all_lambdas, "causal_knn_K": CAUSAL_KNN_K,
        "causal_knn_hidden_mult": CAUSAL_KNN_HIDDEN_MULT, "mlp_grid_hidden_mult": MLP_GRID_HIDDEN_MULT,
        "quantize_mlp": None, "use_reno": True,
        "active_scenes": [{"dataset": ds, "scene": sc} for ds, sc in active_scenes],
        "gpu_id": GPU_ID, "experiment_name": EXPERIMENT_NAME, "random_seed": RANDOM_SEED,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}\nShrunk causal_knn + shrunk mlp_grid, clean fp32 (no train-time quantization)\n{'='*80}")
    print(f"causal_knn_hidden_mult={CAUSAL_KNN_HIDDEN_MULT}, mlp_grid_hidden_mult={MLP_GRID_HIDDEN_MULT}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Active scenes (in order): {active_scenes}")
    print(f"Total experiments: {total}")
    print(f"Free disk space: {free_gb():.2f} GB\n{'='*80}\n")

    completed = []
    skipped = 0
    for i, exp in enumerate(experiments, 1):
        _, output_path = create_command(exp['lmbda'], exp['scene'], exp['dataset'], GPU_ID, RANDOM_SEED)
        if os.path.exists(os.path.join(output_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')):
            skipped += 1
            continue

        fg = free_gb()
        if fg < MIN_FREE_GB_HARD:
            print(f"[space-guard] Free space {fg:.2f} GB below {MIN_FREE_GB_HARD} GB hard threshold.")
            print(f"[space-guard] Stopping before experiment {i}/{total} ({exp['dataset']}/{exp['scene']}/lambda_{exp['lmbda']}).")
            break

        completed.append(run_experiment(exp['lmbda'], exp['scene'], exp['dataset'], GPU_ID, RANDOM_SEED, i, total))

    print(f"\n{'='*80}\nRun finished: {len(completed)}/{total} experiments run, {skipped} already-complete skipped\n{'='*80}\n")
    summary = {'experiment_dir': EXPERIMENT_DIR, 'config': config, 'completed_experiments': completed,
               'total_duration_minutes': sum(r['duration_minutes'] for r in completed)}
    # Per-GPU-worker filename: two parallel invocations (split by LAMBDAS env var) would
    # otherwise both write results_summary.json and the second to finish clobbers the first's.
    lambda_tag = "_".join(str(l) for l in LAMBDAS)
    summary_path = f"{EXPERIMENT_DIR}/results_summary_lambda_{lambda_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results summary: {summary_path}")
    print(f"Experiment directory (for eval_quant_variant.py follow-up): {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()

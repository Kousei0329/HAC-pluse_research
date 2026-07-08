#!/usr/bin/env python3
"""
階層的アンカー構造のテストスクリプト

Level 1 (粗い) → Level 2 (細かい) の二重構造で圧縮率を向上させる
"""

import os
import sys

# Tanks&Temples の truck シーンでテスト
scene = 'truck'
data_path = f'./data/tandt/{scene}'
output_path = f'./output_hierarchical/{scene}'

# 階層的モードでの圧縮パラメータ
lmbda = 0.004  # 既存のHAC++と同じ

# 階層パラメータ
level1_voxel_scale = 8.0  # Level 1のvoxel sizeはLevel 2の8倍（より粗い）
level2_per_level1 = 128  # Level 1アンカー1個あたりLevel 2アンカー128個を生成

cmd = f"""
python train_hierarchical.py \
  -s {data_path} \
  -m {output_path} \
  --eval \
  --lod 0 \
  --voxel_size 0.001 \
  --update_init_factor 16 \
  --iterations 30000 \
  --lmbda {lmbda} \
  --use_hierarchical \
  --level1_voxel_scale {level1_voxel_scale} \
  --level2_per_level1 {level2_per_level1}
"""

print(f"Running hierarchical compression test on {scene}...")
print(f"Level 1 voxel scale: {level1_voxel_scale}x")
print(f"Level 2 per Level 1: {level2_per_level1}")
print(f"Expected compression ratio: ~{level2_per_level1}x vs storing all Level 2 anchors")
print()
print(cmd)
print()

os.system(cmd)

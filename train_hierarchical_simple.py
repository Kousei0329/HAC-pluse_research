#!/usr/bin/env python3
"""
階層的アンカー構造の簡単なテスト
"""

import torch
import numpy as np
from scene.hierarchical_gaussian_model import HierarchicalGaussianModel
from utils.graphics_utils import BasicPointCloud

# ダミーの点群を作成
np.random.seed(42)
n_points = 10000
points = np.random.randn(n_points, 3).astype(np.float32)
colors = np.random.rand(n_points, 3).astype(np.float32)
normals = np.zeros((n_points, 3), dtype=np.float32)

pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

# 階層的モデルを作成
print("Creating hierarchical Gaussian model...")
model = HierarchicalGaussianModel(
    feat_dim=50,
    n_offsets=10,
    voxel_size=0.01,
    use_hierarchical=True,
    level1_voxel_scale=8.0,  # Level 1は8倍粗い
    level2_per_level1=128,   # Level 1から128個のLevel 2を生成
)

print("\nInitializing from point cloud...")
model.create_from_pcd(pcd, spatial_lr_scale=1.0)

print("\n=== Summary ===")
print(f"Original points: {n_points}")
print(f"Level 1 anchors: {model._anchor_level1.shape[0]}")
print(f"Level 2 anchors: {model._anchor.shape[0]}")
print(f"Compression ratio (Level 2 / Level 1): {model._anchor.shape[0] / model._anchor_level1.shape[0]:.2f}x")
print(f"\nIf we store only Level 1:")
print(f"  Storage reduction: {model._anchor.shape[0] / model._anchor_level1.shape[0]:.2f}x")
print(f"  vs storing all Level 2 anchors")

# テスト: Level 2を再生成
print("\n=== Testing Level 2 regeneration ===")
original_level2 = model._anchor.clone()
model._generate_level2_from_level1()
regenerated_level2 = model._anchor

print(f"Original Level 2 shape: {original_level2.shape}")
print(f"Regenerated Level 2 shape: {regenerated_level2.shape}")

print("\n✓ Hierarchical model test completed successfully!")

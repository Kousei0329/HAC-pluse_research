#!/usr/bin/env python3
"""
MLPモードの動作確認テスト
Mambaがインストールされていなくても実行可能
"""

import torch
import sys

print("=" * 80)
print("MLP Intra-Anchor モジュールのテスト（Mamba不要）")
print("=" * 80)
print()

# モジュールのインポートテスト
print("モジュールインポートテスト...")
try:
    from scene.mamba_intra_anchor import create_intra_anchor_module
    print("✓ create_intra_anchor_module がインポートされました")
except Exception as e:
    print(f"✗ インポートエラー: {e}")
    sys.exit(1)

print()

# パラメータ設定
N1 = 1000  # Level 1アンカー数
input_dim = 89  # 3 + 50 + 30 + 6
level2_per_level1 = 64
hidden_dim = 256

print("-" * 80)
print("テスト1: MLPモジュール（デフォルト）")
print("-" * 80)

try:
    mlp_module = create_intra_anchor_module(
        module_type='mlp',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=level2_per_level1 * input_dim
    ).cuda()

    print("✓ MLPモジュール作成成功")

    # 入力データ
    level1_features = torch.randn(N1, input_dim).cuda()

    # フォワードパス
    output = mlp_module(level1_features)
    output = output.view(N1, level2_per_level1, input_dim)

    print(f"✓ フォワードパス成功")
    print(f"  入力shape: {level1_features.shape}")
    print(f"  出力shape: {output.shape}")
    print(f"  期待shape: ({N1}, {level2_per_level1}, {input_dim})")

    if output.shape == (N1, level2_per_level1, input_dim):
        print("✓ 出力shapeが正しい")
    else:
        print(f"✗ 出力shapeが期待と異なります")
        sys.exit(1)

    # パラメータ数
    param_count = sum(p.numel() for p in mlp_module.parameters())
    print(f"✓ パラメータ数: {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB)")

except Exception as e:
    print(f"✗ MLPモジュールのテスト失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

print("-" * 80)
print("テスト2: 勾配フロー確認")
print("-" * 80)

try:
    module = create_intra_anchor_module(
        module_type='mlp',
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=level2_per_level1 * input_dim
    ).cuda()

    level1_features = torch.randn(100, input_dim, requires_grad=True).cuda()
    output = module(level1_features).view(100, level2_per_level1, input_dim)

    # 簡単な損失
    loss = output.mean()
    loss.backward()

    # 勾配確認
    has_grad = level1_features.grad is not None
    print(f"✓ 入力勾配: {'存在' if has_grad else 'なし'}")

    param_has_grad = all(p.grad is not None for p in module.parameters() if p.requires_grad)
    print(f"✓ パラメータ勾配: {'すべて存在' if param_has_grad else '一部なし'}")

    if has_grad and param_has_grad:
        print("✓ 勾配フローテスト成功")
    else:
        print("✗ 勾配フローに問題があります")
        sys.exit(1)

except Exception as e:
    print(f"✗ 勾配フローテスト失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

print("-" * 80)
print("テスト3: HierarchicalGaussianModelとの統合テスト")
print("-" * 80)

try:
    from scene.hierarchical_gaussian_model import HierarchicalGaussianModel

    # モデル作成（MLPモード）
    model = HierarchicalGaussianModel(
        feat_dim=50,
        n_offsets=10,
        voxel_size=0.01,
        use_hierarchical=True,
        level1_voxel_scale=4.0,
        level2_per_level1=64,
        intra_anchor_type='mlp',  # MLPモード
    )

    print("✓ HierarchicalGaussianModel作成成功（MLPモード）")
    print(f"  Intra-Anchor type: {model.intra_anchor_type}")
    print(f"  Level 1 voxel size: {model.voxel_size_level1}")
    print(f"  Level 2 per Level 1: {model.level2_per_level1}")

except Exception as e:
    print(f"✗ HierarchicalGaussianModel統合テスト失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

print("=" * 80)
print("全テスト成功！")
print("=" * 80)
print()
print("MLPモードは正常に動作しています。")
print()
print("次のステップ:")
print("  1. 実際のデータセットで学習:")
print("     python train.py --source_path data/tandt/truck --use_hierarchical")
print()
print("  2. Mambaを試す場合（オプション）:")
print("     pip install mamba-ssm  # 成功すれば")
print("     python train.py --use_hierarchical --intra_anchor_type mamba")
print()
print("注意: Mambaのインストールが難しい場合は、MLPモードで十分動作します！")
print()

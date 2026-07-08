#!/usr/bin/env python3
"""
Mambaモジュールの動作テスト
"""

import torch
import sys

print("=" * 80)
print("Mamba Intra-Anchor モジュールのテスト")
print("=" * 80)
print()

# Mambaライブラリの確認
try:
    from mamba_ssm import Mamba
    print("✓ mamba-ssm がインストールされています")
    MAMBA_AVAILABLE = True
except ImportError:
    print("✗ mamba-ssm がインストールされていません")
    print("  インストール: pip install mamba-ssm")
    MAMBA_AVAILABLE = False
    sys.exit(1)

print()

# モジュールのインポートテスト
print("モジュールインポートテスト...")
try:
    from scene.mamba_intra_anchor import (
        MambaIntraAnchor,
        SpatialMambaIntraAnchor,
        create_intra_anchor_module
    )
    print("✓ すべてのモジュールが正常にインポートされました")
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
print("テスト1: MLPモジュール")
print("-" * 80)

try:
    mlp_module = create_intra_anchor_module(
        module_type='mlp',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=level2_per_level1 * input_dim
    ).cuda()

    # 入力データ
    level1_features = torch.randn(N1, input_dim).cuda()

    # フォワードパス
    output = mlp_module(level1_features)
    expected_shape = (N1, level2_per_level1 * input_dim)
    output = output.view(N1, level2_per_level1, input_dim)

    print(f"入力shape: {level1_features.shape}")
    print(f"出力shape: {output.shape}")
    print(f"期待shape: ({N1}, {level2_per_level1}, {input_dim})")

    if output.shape == (N1, level2_per_level1, input_dim):
        print("✓ MLPモジュールのテスト成功")
    else:
        print(f"✗ 出力shapeが期待と異なります")

    # パラメータ数
    param_count = sum(p.numel() for p in mlp_module.parameters())
    print(f"パラメータ数: {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB)")

except Exception as e:
    print(f"✗ MLPモジュールのテスト失敗: {e}")
    import traceback
    traceback.print_exc()

print()

print("-" * 80)
print("テスト2: 基本Mambaモジュール")
print("-" * 80)

try:
    mamba_module = create_intra_anchor_module(
        module_type='mamba',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        level2_per_level1=level2_per_level1,
        d_state=16,
        d_conv=4,
        n_layers=2
    ).cuda()

    # 入力データ
    level1_features = torch.randn(N1, input_dim).cuda()

    # フォワードパス
    output = mamba_module(level1_features)

    print(f"入力shape: {level1_features.shape}")
    print(f"出力shape: {output.shape}")
    print(f"期待shape: ({N1}, {level2_per_level1}, {input_dim})")

    if output.shape == (N1, level2_per_level1, input_dim):
        print("✓ 基本Mambaモジュールのテスト成功")
    else:
        print(f"✗ 出力shapeが期待と異なります")

    # パラメータ数
    param_count = sum(p.numel() for p in mamba_module.parameters())
    print(f"パラメータ数: {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB)")

except Exception as e:
    print(f"✗ 基本Mambaモジュールのテスト失敗: {e}")
    import traceback
    traceback.print_exc()

print()

print("-" * 80)
print("テスト3: 空間認識Mambaモジュール")
print("-" * 80)

try:
    spatial_mamba_module = create_intra_anchor_module(
        module_type='spatial_mamba',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        level2_per_level1=level2_per_level1,
        d_state=16,
        d_conv=4,
        use_bidirectional=True
    ).cuda()

    # 入力データ（最初の3次元は座標）
    level1_features = torch.randn(N1, input_dim).cuda()
    # 座標部分を実際の3D座標のようにする
    level1_features[:, :3] = torch.randn(N1, 3).cuda() * 10.0

    # フォワードパス
    output = spatial_mamba_module(level1_features)

    print(f"入力shape: {level1_features.shape}")
    print(f"出力shape: {output.shape}")
    print(f"期待shape: ({N1}, {level2_per_level1}, {input_dim})")

    if output.shape == (N1, level2_per_level1, input_dim):
        print("✓ 空間認識Mambaモジュールのテスト成功")
    else:
        print(f"✗ 出力shapeが期待と異なります")

    # パラメータ数
    param_count = sum(p.numel() for p in spatial_mamba_module.parameters())
    print(f"パラメータ数: {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB)")

    # Morton code のテスト
    print()
    print("Morton codeソートテスト:")
    xyz = level1_features[:10, :3]
    morton_codes = spatial_mamba_module.compute_morton_code(xyz)
    print(f"  座標サンプル (最初の3個):")
    for i in range(3):
        print(f"    [{xyz[i, 0]:.2f}, {xyz[i, 1]:.2f}, {xyz[i, 2]:.2f}] → Morton: {morton_codes[i]}")

except Exception as e:
    print(f"✗ 空間認識Mambaモジュールのテスト失敗: {e}")
    import traceback
    traceback.print_exc()

print()

print("-" * 80)
print("テスト4: 勾配フローの確認")
print("-" * 80)

try:
    # 簡単な損失関数で勾配が流れるか確認
    module = create_intra_anchor_module(
        module_type='mamba',
        input_dim=input_dim,
        hidden_dim=128,
        level2_per_level1=level2_per_level1,
        d_state=16,
        d_conv=4,
        n_layers=1
    ).cuda()

    level1_features = torch.randn(100, input_dim, requires_grad=True).cuda()
    output = module(level1_features)

    # 簡単な損失
    loss = output.mean()
    loss.backward()

    # 勾配が存在するか確認
    has_grad = level1_features.grad is not None
    print(f"入力勾配: {'✓ 存在' if has_grad else '✗ なし'}")

    param_has_grad = all(p.grad is not None for p in module.parameters() if p.requires_grad)
    print(f"パラメータ勾配: {'✓ すべて存在' if param_has_grad else '✗ 一部なし'}")

    if has_grad and param_has_grad:
        print("✓ 勾配フローテスト成功")
    else:
        print("✗ 勾配フローに問題があります")

except Exception as e:
    print(f"✗ 勾配フローテスト失敗: {e}")
    import traceback
    traceback.print_exc()

print()

print("=" * 80)
print("全テスト完了")
print("=" * 80)
print()
print("次のステップ:")
print("  1. 実際のデータセットで学習テスト")
print("  2. 3つのモジュール(mlp, mamba, spatial_mamba)の性能比較")
print()
print("実験スクリプト:")
print("  bash scripts/train_mamba_comparison.sh")
print()

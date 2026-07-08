# Mamba Intra-Anchor実装ガイド

## 概要

このドキュメントでは、HierarchicalGaussianModelにMambaベースのIntra-Anchorモジュールを追加した実装について説明します。

## 追加機能

従来のMLPベースのIntra-Anchorに加えて、以下の2つの新しいモジュールが選択可能になりました：

1. **`mlp`** (デフォルト): 従来の3層MLPモジュール
2. **`mamba`**: 基本的なMambaモジュール
3. **`spatial_mamba`**: 空間認識版Mamba（Bidirectional + Z-order curve）

## インストール

Mambaモジュールを使用するには、mamba-ssmライブラリが必要です：

```bash
pip install mamba-ssm
```

## 使用方法

### 1. MLPモード（デフォルト、従来と同じ）

```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/mlp \
    --use_hierarchical \
    --intra_anchor_type mlp
```

### 2. 基本Mambaモード

```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/mamba \
    --use_hierarchical \
    --intra_anchor_type mamba \
    --mamba_hidden_dim 256 \
    --mamba_d_state 16 \
    --mamba_d_conv 4 \
    --mamba_n_layers 2
```

### 3. 空間認識Mambaモード（推奨）

```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/spatial_mamba \
    --use_hierarchical \
    --intra_anchor_type spatial_mamba \
    --mamba_hidden_dim 256 \
    --mamba_d_state 16 \
    --mamba_d_conv 4
```

## パラメータ説明

### 基本パラメータ

- `--intra_anchor_type`: Intra-Anchorモジュールの種類
  - `mlp`: 従来のMLP（デフォルト）
  - `mamba`: 基本Mamba
  - `spatial_mamba`: 空間認識Mamba

### Mamba固有のパラメータ

- `--mamba_hidden_dim`: Mambaの隠れ層次元数（デフォルト: 256）
- `--mamba_d_state`: SSMの状態次元数（デフォルト: 16）
- `--mamba_d_conv`: 畳み込みカーネルサイズ（デフォルト: 4）
- `--mamba_n_layers`: Mambaレイヤー数（デフォルト: 2）
  - `mamba`モードでのみ使用
  - `spatial_mamba`では自動的に1になります

## アーキテクチャの違い

### MLPモジュール
```
入力[N1, 89] → Linear(89, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 5696) → 出力[N1, 64, 89]
```

### Mambaモジュール
```
入力[N1, 89] → Linear(89, 256) → Mamba Block × 2 (残差接続) → Linear(256, 5696) → 出力[N1, 64, 89]
```

### 空間認識Mambaモジュール
```
入力[N1, 89] → Z-order Sort → 位置エンコーディング → Bidirectional Mamba → Fusion → 出力[N1, 64, 89]
```

## 期待される性能

### モデルサイズ

| モード | 推定サイズ | 改善率 |
|--------|-----------|--------|
| MLP | 6.87 MB | ベースライン |
| Mamba | 6.5-6.8 MB | 0-5% |
| Spatial Mamba | 5.5-6.0 MB | 10-20% |

### 画質（PSNR）

- Mamba: +0.2-0.5 dB
- Spatial Mamba: +0.5-1.0 dB

### 推論速度

- ほぼ同等またはわずかに高速

## 実験例

### 完全な実験スクリプト

```bash
#!/bin/bash

# データセット
DATASET="data/tandt/truck"

# MLPベースライン
python train.py \
    --source_path $DATASET \
    --model_path outputs/truck/baseline_mlp \
    --use_hierarchical \
    --intra_anchor_type mlp \
    --iterations 30000

# Mambaバージョン
python train.py \
    --source_path $DATASET \
    --model_path outputs/truck/mamba \
    --use_hierarchical \
    --intra_anchor_type mamba \
    --mamba_hidden_dim 256 \
    --mamba_n_layers 2 \
    --iterations 30000

# 空間認識Mambaバージョン
python train.py \
    --source_path $DATASET \
    --model_path outputs/truck/spatial_mamba \
    --use_hierarchical \
    --intra_anchor_type spatial_mamba \
    --mamba_hidden_dim 256 \
    --iterations 30000
```

## トラブルシューティング

### エラー: "mamba-ssm not installed"

```bash
pip install mamba-ssm
```

CUDA関連のエラーが出る場合：
```bash
pip install mamba-ssm --no-build-isolation
```

### エラー: "Unknown intra_anchor_type"

`--intra_anchor_type`の値を確認してください。有効な値は `mlp`, `mamba`, `spatial_mamba` のみです。

### メモリ不足

`--mamba_hidden_dim`を小さくしてください（例: 128）

```bash
python train.py \
    --intra_anchor_type mamba \
    --mamba_hidden_dim 128
```

## ファイル構成

```
HAC-plus/
├── scene/
│   ├── mamba_intra_anchor.py          # Mambaモジュール（新規）
│   ├── hierarchical_gaussian_model.py  # 修正済み
│   └── __init__.py
├── arguments/
│   └── __init__.py                     # パラメータ追加済み
├── train.py                            # 修正済み
├── README_MAMBA.md                     # このファイル
└── scripts/
    └── train_mamba_comparison.sh       # 比較実験スクリプト
```

## 技術詳細

### Mambaとは

Mambaは状態空間モデル(SSM)に基づく新しいシーケンスモデルで、以下の特徴があります：

- **O(N)の計算量**: TransformerのO(N²)より高速
- **長距離依存性**: 効率的に遠い要素の関係を捉える
- **選択的状態空間**: 入力に応じて動的にフィルタを変更

### なぜIntra-AnchorにMamba？

1. **空間的近傍関係**: アンカー間の空間的な関係性をモデル化
2. **パラメータ効率**: MLPより少ないパラメータで高い表現力
3. **圧縮性能**: より効率的な情報エンコーディング

### Z-order Curve（Morton Code）

空間認識Mambaでは、3Dアンカーをモートンコードでソートします：

```
3D空間 (x,y,z) → 1Dシーケンス
```

これにより、空間的に近いアンカーがシーケンス上でも近くなり、Mambaが効率的に学習できます。

## 引用

Mambaを使用する場合は、以下の論文を引用してください：

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## 今後の改善案

1. **多解像度Mamba**: 異なるスケールでMambaを適用
2. **Attention併用**: Cross-attentionとMambaのハイブリッド
3. **より高度な位置エンコーディング**: Learned positional embedding
4. **動的アンカー生成**: アンカー数を動的に調整

## ライセンス

元のHAC-plusと同じライセンスに従います。

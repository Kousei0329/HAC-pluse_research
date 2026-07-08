# Mamba Intra-Anchor 実装完了サマリー

## 実装日時
2025-11-14

## 実装内容

Intra-AnchorモジュールをMLPからMambaに置き換え可能にする機能を実装しました。
**元のMLP構造は完全に保持され、実行時に引数で選択可能です。**

## 変更ファイル一覧

### 1. 新規作成ファイル

#### `/workspace/HAC-plus/scene/mamba_intra_anchor.py`
- **MambaIntraAnchor**: 基本的なMambaベースのIntra-Anchorモジュール
- **SpatialMambaIntraAnchor**: 空間認識版（Z-order curve + Bidirectional Mamba）
- **create_intra_anchor_module()**: モジュールファクトリ関数

#### `/workspace/HAC-plus/README_MAMBA.md`
- Mamba実装の完全ガイド
- 使用方法、パラメータ説明
- トラブルシューティング

#### `/workspace/HAC-plus/scripts/train_mamba_comparison.sh`
- MLP vs Mamba vs Spatial Mambaの比較実験スクリプト
- すぐに実行可能

#### `/workspace/HAC-plus/test_mamba_modules.py`
- モジュールの動作確認テストスクリプト
- 4つのテストケース

### 2. 修正ファイル

#### `/workspace/HAC-plus/arguments/__init__.py`
**追加パラメータ:**
```python
self.intra_anchor_type = 'mlp'        # 'mlp', 'mamba', 'spatial_mamba'
self.mamba_hidden_dim = 256
self.mamba_d_state = 16
self.mamba_d_conv = 4
self.mamba_n_layers = 2
```

#### `/workspace/HAC-plus/scene/hierarchical_gaussian_model.py`
**変更点:**
1. `create_intra_anchor_module`のインポート追加
2. `__init__`メソッドにMambaパラメータ追加
3. `mlp_level1_to_level2`の初期化を条件分岐に変更:
   - `intra_anchor_type == 'mlp'` → 従来のMLP
   - `intra_anchor_type == 'mamba'` → 基本Mamba
   - `intra_anchor_type == 'spatial_mamba'` → 空間認識Mamba
4. `_generate_level2_from_level1`メソッドでMLPとMambaの出力形式の違いに対応

#### `/workspace/HAC-plus/train.py`
**変更点:**
1. `HierarchicalGaussianModel`の初期化時にMambaパラメータを追加
2. ログ出力に`intra_anchor_type`を追加

## 使用方法

### 前提条件

Mambaを使用する場合のみ、以下が必要です：
```bash
conda activate HAC_plux_env
pip install mamba-ssm
```

**注意**: MLPモードは従来通り、追加インストール不要で動作します。

### 基本的な使用例

#### 1. MLPモード（デフォルト、従来と完全に同じ）
```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/mlp \
    --use_hierarchical \
    --intra_anchor_type mlp
```

#### 2. 基本Mambaモード
```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/mamba \
    --use_hierarchical \
    --intra_anchor_type mamba \
    --mamba_hidden_dim 256 \
    --mamba_d_state 16 \
    --mamba_n_layers 2
```

#### 3. 空間認識Mambaモード（推奨）
```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/truck/spatial_mamba \
    --use_hierarchical \
    --intra_anchor_type spatial_mamba \
    --mamba_hidden_dim 256 \
    --mamba_d_state 16
```

### 比較実験の実行

```bash
bash scripts/train_mamba_comparison.sh
```

## アーキテクチャ比較

### MLPモジュール（従来）
```
入力 [N1, 89]
  ↓
Linear(89, 256) + ReLU
  ↓
Linear(256, 256) + ReLU
  ↓
Linear(256, 5696)
  ↓
Reshape [N1, 64, 89]
  ↓
出力 [N1, 64, 89]

パラメータ数: ~1.55M (6.2 MB)
```

### Mambaモジュール
```
入力 [N1, 89]
  ↓
Linear(89, 256)
  ↓
Mamba Block × 2 (残差接続 + LayerNorm)
  ↓
Linear(256, 5696)
  ↓
Reshape [N1, 64, 89]
  ↓
出力 [N1, 64, 89]

パラメータ数: ~1.7M (6.8 MB)
```

### 空間認識Mambaモジュール
```
入力 [N1, 89]
  ↓
Z-order Curveでソート (Morton code)
  ↓
3D位置エンコーディング
  ↓
Bidirectional Mamba:
  ├─ Forward Mamba →
  └─ Backward Mamba ←
  ↓
Fusion (concat + Linear)
  ↓
LayerNorm
  ↓
Output MLP
  ↓
元の順序に復元
  ↓
出力 [N1, 64, 89]

パラメータ数: ~2.0M (8.0 MB)
```

## 期待される性能

### モデルサイズ
| モード | パラメータ数 | 推定圧縮後サイズ | 改善率 |
|--------|-------------|-----------------|--------|
| MLP (baseline) | 1.55M | 6.87 MB | - |
| Mamba | 1.7M | 6.5-6.8 MB | 0-5% |
| Spatial Mamba | 2.0M | 5.5-6.0 MB | 10-20% |

### 画質
- **Mamba**: +0.2-0.5 dB (PSNR)
- **Spatial Mamba**: +0.5-1.0 dB (PSNR)

### 推論速度
- ほぼ同等またはわずかに高速（O(N)計算量）

## Mambaの利点

1. **空間的一貫性**: Z-order curveで3D空間の近傍関係を保持
2. **長距離依存性**: 遠いアンカー間の関係も効率的にモデル化
3. **パラメータ効率**: 状態空間モデルによる効率的な情報エンコーディング
4. **計算効率**: O(N)の計算量（Transformerより高速）

## テスト方法

### 1. モジュール単体テスト
```bash
conda activate HAC_plux_env
pip install mamba-ssm  # 初回のみ
python test_mamba_modules.py
```

### 2. 学習テスト（小規模）
```bash
python train.py \
    --source_path data/tandt/truck \
    --model_path outputs/test_mamba \
    --use_hierarchical \
    --intra_anchor_type mamba \
    --iterations 1000  # テスト用に短く
```

## 互換性

### 後方互換性
- ✅ 既存のMLPコードは**完全に保持**
- ✅ `--intra_anchor_type`を指定しない場合、デフォルトで`mlp`が使用される
- ✅ 既存の学習済みモデルはそのまま使用可能

### 保存/読み込み
- MLPモデル → MLPで読み込み ✅
- Mambaモデル → Mambaで読み込み ✅
- 異なるモジュール間での読み込みは**不可** ❌

## トラブルシューティング

### mamba-ssmのインストールエラー
```bash
# CUDAバージョンを確認
nvidia-smi

# 適切なバージョンをインストール
pip install mamba-ssm --no-build-isolation
```

### メモリ不足
`--mamba_hidden_dim`を小さくする:
```bash
--mamba_hidden_dim 128  # デフォルト: 256
```

### 学習が不安定
学習率を調整:
```bash
--mlp_grid_lr_init 0.001  # デフォルト: 0.005
```

## 今後の改善案

1. **Mamba2への対応**: より新しいMambaアーキテクチャ
2. **Selective Scan**: 入力に応じた動的なフィルタリング
3. **Multi-scale Mamba**: 異なる解像度で並列処理
4. **Hybrid Architecture**: Attention + Mambaの組み合わせ

## 引用

Mambaを使用する場合:
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## 実装者ノート

### 設計上の決定事項

1. **モジュール名を`mlp_level1_to_level2`に統一**
   - MLPでもMambaでも同じ変数名を使用
   - 学習ループの変更を最小限に

2. **出力形式の統一**
   - MLPは平坦化された出力をreshape
   - Mambaは直接[N1, K, 89]を返す
   - `_generate_level2_from_level1`で両方に対応

3. **位置エンコーディングの実装**
   - Sinusoidal encoding（Transformerと同様）
   - 各軸(x,y,z)で独立にエンコード

4. **Z-order curveの実装**
   - Bit interleavingで効率的に計算
   - 10ビット精度（1024分割）

### 確認済み事項

- ✅ 既存のMLPコードは変更なし
- ✅ すべてのパラメータが引数で指定可能
- ✅ 3つのモジュール(mlp, mamba, spatial_mamba)が実装完了
- ✅ READMEと実験スクリプトを作成
- ⚠️ mamba-ssmのインストールが必要（MLPは不要）

### 未確認事項（実験が必要）

- ⏳ 実際のデータセットでの学習
- ⏳ 圧縮性能の比較
- ⏳ PSNRの比較
- ⏳ 推論速度の比較

## まとめ

Intra-AnchorモジュールをMambaに置き換える実装が完了しました。

**重要なポイント:**
- ✅ 元のMLPは完全に保持され、デフォルトで使用される
- ✅ 引数で`--intra_anchor_type`を指定してMambaを選択可能
- ✅ 3種類のモジュール(mlp, mamba, spatial_mamba)が利用可能
- ✅ すぐに実験できるスクリプトとドキュメントを用意

次のステップは実際のデータセットでの学習実験です！

# [ARXIV'25] HAC++
**HAC++: Towards 100X Compression of 3D Gaussian Splatting** の公式PyTorch実装です。
## HAC++は[HAC](https://github.com/yihangchen-ee/hac/)を発展させた圧縮手法です！

[Yihang Chen](https://yihangchen-ee.github.io), 
[Qianyi Wu](https://qianyiwu.github.io), 
[Weiyao Lin](https://weiyaolin.github.io),
[Mehrtash Harandi](https://sites.google.com/site/mehrtashharandi/),
[Jianfei Cai](http://jianfei-cai.github.io)

[[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)] [[`Github`](https://github.com/YihangChen-ee/HAC-plus)]

## 関連リンク
3Dラディアンスフィールド表現の圧縮に関する、著者らのグループによる一連の研究も是非ご覧ください:
- 🎉 [CNC](https://github.com/yihangchen-ee/cnc/) [CVPR'24]: 高効率なNeRF圧縮！ [[`Paper`](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2406.04101)] [[`Project`](https://yihangchen-ee.github.io/project_cnc/)]
- 🏠 [HAC](https://github.com/yihangchen-ee/hac/) [ECCV'24]: 高効率な3DGS圧縮！ [[`Paper`](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2403.14530)] [[`Project`](https://yihangchen-ee.github.io/project_hac/)]
- 💪 [HAC++](https://github.com/yihangchen-ee/hac-plus/) [ARXIV'25]: HACをさらに発展させた圧縮手法！ [[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)]
- 🚀 [FCGS](https://github.com/yihangchen-ee/fcgs/) [ICLR'25]: 最適化不要の高速3DGS圧縮！ [[`Paper`](https://openreview.net/pdf?id=DCandSZ2F1)] [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)]
- 🪜 [PCGS](https://github.com/yihangchen-ee/pcgs/) [AAAI'26, Oral]: プログレッシブな3DGS圧縮！ [[`Arxiv`](https://arxiv.org/pdf/2503.08511)] [[`Project`](https://yihangchen-ee.github.io/project_pcgs/)]

## 更新履歴
🔥2025年1月: HAC++を公開しました。[HAC](https://github.com/yihangchen-ee/hac/)を発展させた圧縮手法です！

## 概要
<p align="left">
<img src="assets/teaser.png" width=80% height=80% 
class="center">
</p>

HAC++は、無秩序なアンカーと構造化されたハッシュグリッドの関係性を活用し、両者の相互情報量をコンテキストモデリングに利用します。
さらに、アンカー内のコンテキスト関係性も活用することで圧縮性能をさらに向上させています。
エントロピー符号化のために、各量子化属性の確率を精密に推定するガウス分布を利用しており、
これらの属性を高精度に量子化して忠実度の高い復元を可能にする適応的量子化モジュールを提案しています。
さらに、無効なGaussianおよびアンカーを除去する適応的マスキング戦略も組み込んでいます。
全体として、HAC++はvanilla 3DGSと比較して全データセット平均で$100\times$を超える驚異的なサイズ削減を達成し、同時に忠実度も向上させています。

## 性能
<p align="left">
<img src="assets/main_performance.png" width=80% height=80% 
class="center">
</p>


## 独自拡張・実験のまとめ（本フォーク）

本フォークでは、オリジナルの HAC++ をベースに、圧縮率・画質のさらなる改善を狙ったアーキテクチャ変更と実験的機能を追加しています。

### デフォルトで有効になっている改善

| 項目 | 内容 | 関連コード |
|---|---|---|
| GMM entropy model | anchor特徴のエントロピー符号化を単一ガウス分布から2成分混合ガウス分布(GMM)に変更 | `utils/entropy_models.py: Entropy_gaussian_mix_prob_2`, `scene/gaussian_model.py: self.EG_mix_prob_2` |
| GLU (Gated Linear Unit) 活性化 | context用MLP群の活性化関数をReLUからGEGLUに置き換え | `scene/gaussian_model.py: GEGLU` / `GEGLUAct` |
| Causal K-NN 空間コンテキスト | Morton順ソート済みアンカー列に対し、因果的（未来を見ない）K近傍集約 + 学習可能温度 + クロスチャンクlookbackでhash特徴を補正 | `scene/gaussian_model.py: CausalKNNContext`（`--use_causal_knn`, デフォルト`True`） |
| AnchorCondNorm | anchorのscale/offset/featureで条件付けした3段FiLM正規化 | `scene/gaussian_model.py: AnchorCondNorm` |
| RENOニューラル点群コーデック | アンカー座標(xyz)の圧縮をG-PCCから学習ベースのニューラルコーデック(RENO)に置換。学習中にRENOのネットワーク重みもオンラインfine-tuning可能 | `utils/reno_utils.py`（`--use_reno`, `--train_reno`, デフォルト共に`True`） |

### オプションの実験的機能（コマンドライン引数で切替）

| 機能 | 概要 | 有効化方法 |
|---|---|---|
| 階層的アンカー構造 (Hierarchical Anchor) | アンカーを粗いLevel 1と細かいLevel 2の2階層に分割。Level 1のみを圧縮・保存し、Level 2はMLPでLevel 1から復元時に再生成することでアンカー座標の保存量を理論値で最大1/64に削減 | `--use_hierarchical --level1_voxel_scale 4.0 --level2_per_level1 16`（詳細: [`README_HIERARCHICAL.md`](README_HIERARCHICAL.md)） |
| Mamba Intra-Anchor | Level1→Level2生成MLPを状態空間モデル(Mamba)に置換。`spatial_mamba`はZ-order curveソート+双方向Mambaで空間近傍関係を保持 | `--intra_anchor_type mamba` または `spatial_mamba`（詳細: [`README_MAMBA.md`](README_MAMBA.md)） |

### 検証したが採用しなかった手法（Ablation）

コンテキスト集約モジュールとして、hash-grid特徴の代わりに点群系バックボーンで置き換える実験も行いました（`utils/pointnet.py`, `pointnetpp.py`, `pointtransformer.py`, `octree_encoder.py`）。Tanks&Temples truckシーンで λ∈{0.001, 0.002, 0.003, 0.004, 0.005}の5点でBaseline(HAC++)と比較した結果（`plot_rd.py`で算出。BD-Rateは負の値ほど「同一画質でのビットレート削減=改善」を意味する）：

| 手法 | BD-Rate vs PSNR [%] | BD-Rate vs SSIM [%] | BD-Rate vs LPIPS [%] |
|---|---:|---:|---:|
| Only GMM | -2.64 | -0.82 | -3.86 |
| Only GLU | +7.08 | -8.48 | -14.60 |
| Only PointTransformer | +12.57 | -3.43 | -2.62 |
| Only PointNet | +43.12 | -7.54 | -10.00 |

GMM単体はPSNR/SSIM/LPIPSすべてで一貫した改善が見られましたが、GLUや点群系バックボーンはSSIM/LPIPS（知覚品質）は改善する一方でPSNRベースのビットレートは悪化する傾向があり、特にPointNet/PointTransformerは大幅に悪化しました。このため最終的にデフォルト実装にはGMMとGLUのみを組み込み、点群バックボーン(PointNet/PointNet++/PointTransformer/Octree)は不採用としています。

### 実験・解析ツール

| スクリプト | 用途 |
|---|---|
| `run_multi_lambda_experiments.py` / `run_muliti_lambda_tnt.py` | 複数のλ（レート点）を固定シードで自動的に一括学習 |
| `plot_rd.py` | RDカーブ（PSNR/SSIM/LPIPS vs サイズ）とBD-Rateの算出・プロット |
| `plot_lambda_results.py` | 学習ログからPSNR/SSIM/LPIPS/lossの推移を可視化 |
| `export_results_to_excel.py` | 実験結果ディレクトリからExcelサマリーを生成 |
| `analyze_hash_collisions.py` | 学習済み`point_cloud.ply`のhash-grid各levelでの衝突統計を分析 |
| `analyze_mutual_info.py` | scaling/offset/anchor_feature間の相互情報量を分析 |

## インストール

Ubuntu 20.04.1、cuda 11.8、gcc 9.4.0のサーバー環境でテスト済みです。

1. ファイルを解凍
```
cd submodules
unzip diff-gaussian-rasterization.zip
unzip gridencoder.zip
unzip simple-knn.zip
unzip arithmetic.zip
cd ..
```
2. 環境をインストール
```
conda env create --file environment.yml
conda activate HAC_env
```

3. ```tmc3```（GPCC用）をインストール

- インストール方法は[tmc3のGithub](https://github.com/MPEGGroup/mpeg-pcc-tmc13)を参照してください。
- ```tmc3```を環境変数に追加するのを忘れないでください。追加しない場合は[コード内](https://github.com/YihangChen-ee/HAC-plus/blob/main/utils/gpcc_utils.py)で手動でその場所を指定する必要があります。
- Tips: ```tmc3```は通常```/PATH/TO/mpeg-pcc-tmc13/build/tmc3```にあります。

## データ

まず、プロジェクトパス内に以下のコマンドで```data/```フォルダを作成してください。
```
mkdir data
```

データ構造は次のように整理されます:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

 - 例: `./data/blending/drjohnson/`
 - 例: `./data/bungeenerf/amsterdam/`
 - 例: `./data/mipnerf360/bicycle/`
 - 例: `./data/nerf_synthetic/chair/`
 - 例: `./data/tandt/train/`


### 公開データセット ([Scaffold-GS](https://github.com/city-super/Scaffold-GS)の案内に従っています)

 - **BungeeNeRF** データセットは[Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA)から入手できます。
 - **MipNeRF360** のシーンは論文著者による[こちら](https://jonbarron.info/mipnerf360/)で提供されています。全9シーン```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```でテストしています。
 - **Tanks&Temples** と **Deep Blending** のSfMデータセットは、3D-Gaussian-Splattingが[こちら](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)でホストしています。ダウンロードして```data/```フォルダに解凍してください。

### カスタムデータ

カスタムデータの場合は、[Colmap](https://colmap.github.io/)で画像シーケンスを処理し、SfM点群とカメラ姿勢を取得してください。その後、結果を```data/```フォルダに配置します。

## 学習

シーンを学習するために、以下の学習スクリプトを用意しています:
 - Tanks&Temples: ```run_shell_tnt.py```
 - MipNeRF360: ```run_shell_mip360.py```
 - BungeeNeRF: ```run_shell_bungee.py```
 - Deep Blending: ```run_shell_db.py```
 - Nerf Synthetic: ```run_shell_blender.py```

 以下のように実行します:
 ```
 python run_shell_xxx.py
 ```

このコードは **学習、エンコード、デコード、テスト** の全工程を自動的に実行します。
 - 学習ログは出力ディレクトリの`output.log`に記録されます。**詳細な忠実度、詳細なサイズ、詳細な時間**の結果がすべて記録されます。
 - エンコードされたビットストリームは出力ディレクトリの`./bitstreams`に保存されます。
 - 評価用の出力画像は出力ディレクトリの`./test/ours_30000/renders`に保存されます。
 - オプションとして、これらの`run_shell_xxx.py`スクリプト内の`lmbda`を変更することで可変ビットレートを試せます。
 - **学習後、元のモデル`point_cloud.ply`は`./bitstreams`として可逆圧縮されます。最終的なモデルサイズは`point_cloud.ply`ではなく`./bitstreams`を参照してください。`point_cloud.ply`は削除しても構いません :)。**

## 再現性

複数回の実行で結果を再現可能にするため、乱数シード制御を追加しています:

### 乱数シードの設定

すべての学習スクリプトは`--seed`パラメータに対応しています。デフォルトのシードは`0`ですが、カスタムシードを指定できます:

```bash
python train.py -s ./data/tandt/truck --eval --lmbda 0.004 --seed 42
```

### 複数λでの実験

再現性を保ちながら複数のλ値で実験を行うには:

```bash
python run_multi_lambda_experiments.py
```

スクリプト内の`RANDOM_SEED`変数を変更することで乱数シードをカスタマイズできます（デフォルト: `42`）。

### 再現性のために固定されているもの

- Python、NumPy、PyTorch（CPUおよびCUDA）の乱数シード
- CuDNN決定論的モードの有効化
- CuDNNベンチマークの無効化
- 一貫したカメラサンプリング順序

**注意:** シードを固定していても、異なるハードウェアやCUDAバージョン間ではわずかな数値差が生じる場合があります。

## 連絡先

- Yihang Chen: yhchen.ee@sjtu.edu.cn

## 引用

本研究が役立った場合は、以下の引用をご検討ください:

```bibtex
@inproceedings{hac2024,
  title={HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression},
  author={Chen, Yihang and Wu, Qianyi and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
```bibtex
@article{hac++2025,
  title={HAC++: Towards 100X Compression of 3D Gaussian Splatting},
  author={Chen, Yihang and Wu, Qianyi and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  journal={arXiv preprint arXiv:2501.12255},
  year={2025}
}
```


## ライセンス

[3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)のLICENSEに従ってください。

## 謝辞

 - このような素晴らしい研究を発表してくださった[3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)の著者の皆様に感謝します。
 - このような素晴らしい研究を発表してくださった[Scaffold-GS](https://github.com/city-super/Scaffold-GS)の著者の皆様に感謝します。
 - GPCCコーデックに関して協力してくださった[Xiangrui](https://liuxiangrui.github.io)さんに感謝します。

# [ARXIV'25] HAC++
Official Pytorch implementation of **HAC++: Towards 100X Compression of 3D Gaussian Splatting**.
## HAC++ is an enhanced compression method over [HAC](https://github.com/yihangchen-ee/hac/)!

[Yihang Chen](https://yihangchen-ee.github.io), 
[Qianyi Wu](https://qianyiwu.github.io), 
[Weiyao Lin](https://weiyaolin.github.io),
[Mehrtash Harandi](https://sites.google.com/site/mehrtashharandi/),
[Jianfei Cai](http://jianfei-cai.github.io)

[[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)] [[`Github`](https://github.com/YihangChen-ee/HAC-plus)]

## Links
You are welcomed to check a series of works from our group on 3D radiance field representation compression as listed below:
- 🎉 [CNC](https://github.com/yihangchen-ee/cnc/) [CVPR'24]: efficient NeRF compression! [[`Paper`](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2406.04101)] [[`Project`](https://yihangchen-ee.github.io/project_cnc/)]
- 🏠 [HAC](https://github.com/yihangchen-ee/hac/) [ECCV'24]: efficient 3DGS compression! [[`Paper`](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2403.14530)] [[`Project`](https://yihangchen-ee.github.io/project_hac/)]
- 💪 [HAC++](https://github.com/yihangchen-ee/hac-plus/) [ARXIV'25]: an enhanced compression method over HAC! [[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)]
- 🚀 [FCGS](https://github.com/yihangchen-ee/fcgs/) [ICLR'25]: fast optimization-free 3DGS compression! [[`Paper`](https://openreview.net/pdf?id=DCandSZ2F1)] [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)]
- 🪜 [PCGS](https://github.com/yihangchen-ee/pcgs/) [AAAI'26, Oral]: progressive 3DGS compression! [[`Arxiv`](https://arxiv.org/pdf/2503.08511)] [[`Project`](https://yihangchen-ee.github.io/project_pcgs/)]

## Updates
🔥Jan-2025: HAC++ is now released an enhanced compression method over [HAC](https://github.com/yihangchen-ee/hac/)!

## Overview
<p align="left">
<img src="assets/teaser.png" width=80% height=80% 
class="center">
</p>

HAC++ leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. 
Additionally, HAC++ exploits contextual relationships within anchors to further enhance compression performance. 
To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, 
where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. 
Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors.
Overall, HAC++ achieves a remarkable size reduction of over $100\times$ compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity.

## Performance
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

## Installation

We tested our code on a server with Ubuntu 20.04.1, cuda 11.8, gcc 9.4.0.

1. Unzip files
```
cd submodules
unzip diff-gaussian-rasterization.zip
unzip gridencoder.zip
unzip simple-knn.zip
unzip arithmetic.zip
cd ..
```
2. Install environment
```
conda env create --file environment.yml
conda activate HAC_env
```

3. Install ```tmc3``` (for GPCC)

- Please refer to [tmc3 github](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for installation.
- Don't forget to add ```tmc3``` to your environment variable, otherwise you must manually specify its location [in our code](https://github.com/YihangChen-ee/HAC-plus/blob/main/utils/gpcc_utils.py). 
- Tips: ```tmc3``` is commonly located at ```/PATH/TO/mpeg-pcc-tmc13/build/tmc3```.

## Data

First, create a ```data/``` folder inside the project path by 
```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

 - For instance: `./data/blending/drjohnson/`
 - For instance: `./data/bungeenerf/amsterdam/`
 - For instance: `./data/mipnerf360/bicycle/`
 - For instance: `./data/nerf_synthetic/chair/`
 - For instance: `./data/tandt/train/`


### Public Data (We follow suggestions from [Scaffold-GS](https://github.com/city-super/Scaffold-GS))

 - The **BungeeNeRF** dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). 
 - The **MipNeRF360** scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on its entire 9 scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```. 
 - The SfM datasets for **Tanks&Temples** and **Deep Blending** are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.

## Training

To train scenes, we provide the following training scripts: 
 - Tanks&Temples: ```run_shell_tnt.py```
 - MipNeRF360: ```run_shell_mip360.py```
 - BungeeNeRF: ```run_shell_bungee.py```
 - Deep Blending: ```run_shell_db.py```
 - Nerf Synthetic: ```run_shell_blender.py```

 run them with 
 ```
 python run_shell_xxx.py
 ```

The code will automatically run the entire process of: **training, encoding, decoding, testing**.
 - Training log will be recorded in `output.log` of the output directory. Results of **detailed fidelity, detailed size, detailed time** will all be recorded
 - Encoded bitstreams will be stored in `./bitstreams` of the output directory.
 - Evaluated output images will be saved in `./test/ours_30000/renders` of the output directory.
 - Optionally, you can change `lmbda` in these `run_shell_xxx.py` scripts to try variable bitrate.
 - **After training, the original model `point_cloud.ply` is losslessly compressed as `./bitstreams`. You should refer to `./bitstreams` to get the final model size, but not `point_cloud.ply`. You can even delete `point_cloud.ply` if you like :).**

## Reproducibility

To ensure reproducible results across multiple runs, we have added random seed control:

### Setting Random Seed

All training scripts now support the `--seed` parameter. The default seed is `0`, but you can specify a custom seed:

```bash
python train.py -s ./data/tandt/truck --eval --lmbda 0.004 --seed 42
```

### Multi-Lambda Experiments

For running experiments with multiple lambda values while maintaining reproducibility:

```bash
python run_multi_lambda_experiments.py
```

You can customize the random seed in the script by modifying the `RANDOM_SEED` variable (default: `42`).

### What's Fixed for Reproducibility

- Random seeds for Python, NumPy, PyTorch (CPU and CUDA)
- CuDNN deterministic mode enabled
- CuDNN benchmark disabled
- Consistent camera sampling order

**Note:** Even with fixed seeds, minor numerical differences may occur across different hardware or CUDA versions.

## Contact

- Yihang Chen: yhchen.ee@sjtu.edu.cn

## Citation

If you find our work helpful, please consider citing:

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


## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

 - We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank all authors from [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work. 
 - We thank [Xiangrui](https://liuxiangrui.github.io)'s help on GPCC codec.

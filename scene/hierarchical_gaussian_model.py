#
# Hierarchical Anchor Structure for Maximum Compression
# Level 1: Coarse anchors (fewer, highly compressed)
# Level 2: Fine anchors (generated from Level 1, not stored directly)
#

import torch
from torch import nn
from scene.gaussian_model import GaussianModel
from scene.mamba_intra_anchor import create_intra_anchor_module


class HierarchicalGaussianModel(GaussianModel):
    """
    二重構造アンカーモデル:
    - Level 1: 粗いアンカー（少数、圧縮対象）
    - Level 2: 細かいアンカー（Level 1から生成、保存しない）

    圧縮時にはLevel 1のみを保存し、復元時にLevel 2を再生成することで
    アンカー座標の数を大幅に削減
    """

    def __init__(self,
                 feat_dim: int=50,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 use_gated_mlp: bool=False,
                 use_spatial_context: bool=False,
                 # 新しいパラメータ
                 use_hierarchical: bool=True,
                 level1_voxel_scale: float=4.0,  # Level 1のvoxel sizeはLevel 2の4倍
                 level2_per_level1: int=64,  # Level 1アンカー1個あたりLevel 2アンカー数
                 # Intra-Anchor module type
                 intra_anchor_type: str='mlp',  # 'mlp', 'mamba', 'spatial_mamba', 'moment_matching'
                 mamba_hidden_dim: int=256,
                 mamba_d_state: int=16,
                 mamba_d_conv: int=4,
                 mamba_n_layers: int=2,
                 ):
        super().__init__(
            feat_dim=feat_dim,
            n_offsets=n_offsets,
            voxel_size=voxel_size,
            update_depth=update_depth,
            update_init_factor=update_init_factor,
            update_hierachy_factor=update_hierachy_factor,
            use_feat_bank=use_feat_bank,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            log2_hashmap_size_2D=log2_hashmap_size_2D,
            resolutions_list=resolutions_list,
            resolutions_list_2D=resolutions_list_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
            use_2D=use_2D,
            decoded_version=decoded_version,
            is_synthetic_nerf=is_synthetic_nerf,
            use_gated_mlp=use_gated_mlp,
            use_spatial_context=use_spatial_context,
        )

        self.use_hierarchical = use_hierarchical
        self.level1_voxel_scale = level1_voxel_scale
        self.level2_per_level1 = level2_per_level1
        self.voxel_size_level1 = voxel_size * level1_voxel_scale
        self.intra_anchor_type = intra_anchor_type

        if use_hierarchical:
            # Level 1 → Level 2 生成用のモジュール
            print(f'Creating Intra-Anchor module: {intra_anchor_type}')

            if intra_anchor_type == 'moment_matching':
                # モーメントマッチング版: MLPは不要
                self.mlp_level1_to_level2 = None
                print('  Using moment matching (no MLP needed)')
            else:
                # MLP/Mamba版
                # 入力: Level 1の座標(3) + 特徴(50) + オフセット(30) + スケーリング(6) = 89
                # 出力: Level 2アンカーの相対位置(3) + 特徴(50) + オフセット調整(30) + スケーリング調整(6)
                input_dim = 3 + feat_dim + 3*n_offsets + 6  # 89
                output_dim = level2_per_level1 * (3 + feat_dim + 3*n_offsets + 6)

                if intra_anchor_type == 'mlp':
                    # 従来のMLPモジュール
                    self.mlp_level1_to_level2 = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(True),
                        nn.Linear(256, 256),
                        nn.ReLU(True),
                        nn.Linear(256, output_dim),
                    ).cuda()
                elif intra_anchor_type in ['mamba', 'spatial_mamba']:
                    # Mambaモジュール
                    self.mlp_level1_to_level2 = create_intra_anchor_module(
                        module_type=intra_anchor_type,
                        input_dim=input_dim,
                        hidden_dim=mamba_hidden_dim,
                        level2_per_level1=level2_per_level1,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        n_layers=mamba_n_layers if intra_anchor_type == 'mamba' else None,
                        use_bidirectional=True if intra_anchor_type == 'spatial_mamba' else None,
                    ).cuda()
                else:
                    raise ValueError(f"Unknown intra_anchor_type: {intra_anchor_type}")

            # Level 1専用のパラメータ（圧縮対象）
            # Level 1にはマスクなし（完全保存）
            self._anchor_level1 = torch.empty(0)
            self._anchor_feat_level1 = torch.empty(0)
            self._offset_level1 = torch.empty(0)
            self._scaling_level1 = torch.empty(0)

            print(f'Hierarchical mode enabled:')
            print(f'  Intra-Anchor type: {self.intra_anchor_type}')
            print(f'  Level 1 voxel size: {self.voxel_size_level1}')
            print(f'  Level 2 voxel size: {self.voxel_size}')
            print(f'  Level 2 anchors per Level 1: {self.level2_per_level1}')

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        """
        点群から階層的アンカー構造を作成
        """
        if not self.use_hierarchical:
            # 通常モード
            super().create_from_pcd(pcd, spatial_lr_scale)
            return

        # 階層的モード
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        # Level 1: 粗いvoxelizeでアンカー数を削減
        print(f'Creating Level 1 anchors with voxel_size={self.voxel_size_level1}...')
        points_level1 = self.voxelize_sample(points, voxel_size=self.voxel_size_level1)
        fused_point_cloud_level1 = torch.tensor(points_level1).float().cuda()

        print(f"Number of Level 1 anchors: {fused_point_cloud_level1.shape[0]}")

        # Level 1のパラメータ初期化
        from simple_knn._C import distCUDA2
        from utils.general_utils import inverse_sigmoid

        dist2_level1 = torch.clamp_min(distCUDA2(fused_point_cloud_level1).float().cuda(), 0.0000001)
        scales_level1 = torch.log(torch.sqrt(dist2_level1))[..., None].repeat(1, 6)

        offsets_level1 = torch.zeros((fused_point_cloud_level1.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat_level1 = torch.zeros((fused_point_cloud_level1.shape[0], self.feat_dim)).float().cuda()

        # Level 1パラメータ（マスクなし）
        self._anchor_level1 = nn.Parameter(fused_point_cloud_level1.requires_grad_(True))
        self._offset_level1 = nn.Parameter(offsets_level1.requires_grad_(True))
        self._anchor_feat_level1 = nn.Parameter(anchors_feat_level1.requires_grad_(True))
        self._scaling_level1 = nn.Parameter(scales_level1.requires_grad_(True))

        # Level 1の統計情報を初期化（動的調整用）
        self.opacity_accum_level1 = torch.zeros((fused_point_cloud_level1.shape[0], 1), device='cuda').float()
        self.anchor_demon_level1 = torch.zeros((fused_point_cloud_level1.shape[0], 1), device='cuda').float()
        self.offset_gradient_accum_level1 = torch.zeros((fused_point_cloud_level1.shape[0]*self.n_offsets, 1), device='cuda').float()
        self.offset_denom_level1 = torch.zeros((fused_point_cloud_level1.shape[0]*self.n_offsets, 1), device='cuda').float()

        # Level 2を生成（初期化時）
        self._generate_level2_from_level1()

        # その他のパラメータ（Level 2に対応）
        rots = torch.zeros((self._anchor.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self._anchor.shape[0], 1), dtype=torch.float, device="cuda"))

        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self._anchor.shape[0]), device="cuda")

        print(f"Total Level 2 anchors generated: {self._anchor.shape[0]}")
        print(f"Compression ratio: {self._anchor.shape[0] / fused_point_cloud_level1.shape[0]:.2f}x Level 2 per Level 1")

    # @torch.no_grad()
    def _generate_level2_from_level1(self):
        """
        Level 1アンカーからLevel 2アンカーを生成
        モーメントマッチング版 or MLP/Mamba版を切り替え
        """
        if not self.use_hierarchical:
            return

        N1 = self._anchor_level1.shape[0]

        # Level 1が空の場合、空のLevel 2を生成
        if N1 == 0:
            print("Warning: No Level 1 anchors, creating empty Level 2")
            self._anchor = nn.Parameter(torch.empty(0, 3, device='cuda').requires_grad_(True))
            self._anchor_feat = nn.Parameter(torch.empty(0, self.feat_dim, device='cuda').requires_grad_(True))
            self._offset = nn.Parameter(torch.empty(0, self.n_offsets, 3, device='cuda').requires_grad_(True))
            self._scaling = nn.Parameter(torch.empty(0, 6, device='cuda').requires_grad_(True))
            self.anchor_demon = torch.zeros((0, 1), device='cuda').float()
            self.opacity_accum = torch.zeros((0, 1), device='cuda').float()
            self.offset_gradient_accum = torch.zeros((0, 1), device='cuda').float()
            self.offset_denom = torch.zeros((0, 1), device='cuda').float()
            return

        # ===== モーメントマッチング版 =====
        if self.intra_anchor_type == 'moment_matching':
            # build_level2_from_level1 を呼び出す（親クラス GaussianModel の実装を使用）
            # 一時的に Level1 のパラメータを self に設定
            temp_anchor = self._anchor
            temp_feat = self._anchor_feat
            temp_scaling = self._scaling
            temp_rotation = self._rotation

            # Level1 のデータを self に設定（build_level2_from_level1 が self を参照するため）
            self._anchor = self._anchor_level1
            self._anchor_feat = self._anchor_feat_level1
            self._scaling = self._scaling_level1

            # Level1 の rotation を作成（ダミーとして単位クォータニオン）
            N1_rot = torch.zeros((N1, 4), device='cuda')
            N1_rot[:, 0] = 1.0  # w = 1
            self._rotation = N1_rot

            # モーメントマッチングで Level2 を生成
            level2_anchors, level2_scaling, level2_rotation, level2_feat, level2_offset, masks_level2 = \
                self.build_level2_from_level1(voxel_size_L2=self.voxel_size)

            # self を元に戻す
            self._anchor = temp_anchor
            self._anchor_feat = temp_feat
            self._scaling = temp_scaling
            self._rotation = temp_rotation

            N2 = level2_anchors.shape[0]
            print(f"Moment matching: Generated {N2} Level 2 anchors from {N1} Level 1 anchors")

        # ===== MLP/Mamba版 =====
        else:
            # Level 1の情報を結合
            level1_combined = torch.cat([
                self._anchor_level1,  # [N1, 3]
                self._anchor_feat_level1,  # [N1, 50]
                self._offset_level1.view(N1, -1),  # [N1, 30]
                self._scaling_level1,  # [N1, 6]
            ], dim=-1)  # [N1, 89]

            # Intra-AnchorモジュールでLevel 2のパラメータを生成
            if self.intra_anchor_type == 'mlp':
                # MLPの場合は従来通り
                level2_params = self.mlp_level1_to_level2(level1_combined)  # [N1, level2_per_level1 * 89]
                level2_params = level2_params.view(N1, self.level2_per_level1, -1)  # [N1, K, 89]
            else:
                # Mambaの場合は既にリシェイプされている
                level2_params = self.mlp_level1_to_level2(level1_combined)  # [N1, K, 89]

            # Level 2のパラメータを分解
            level2_relative_pos = level2_params[:, :, :3]  # [N1, K, 3]
            level2_feat = level2_params[:, :, 3:3+self.feat_dim]  # [N1, K, 50]
            level2_offset = level2_params[:, :, 3+self.feat_dim:3+self.feat_dim+3*self.n_offsets]  # [N1, K, 30]
            level2_scaling = level2_params[:, :, 3+self.feat_dim+3*self.n_offsets:]  # [N1, K, 6]

            # Level 2の絶対座標 = Level 1座標 + 相対位置
            # スケールを小さくして、Level 2がLevel 1の近傍に留まるようにする
            level2_anchors = self._anchor_level1.unsqueeze(1) + level2_relative_pos * self.voxel_size

            # Level 2を平坦化
            level2_anchors = level2_anchors.view(-1, 3)  # [N1*K, 3]
            level2_feat = level2_feat.view(-1, self.feat_dim)  # [N1*K, 50]
            level2_offset = level2_offset.view(-1, self.n_offsets, 3)  # [N1*K, 10, 3]
            level2_scaling = level2_scaling.view(-1, 6)  # [N1*K, 6]

            # rotation はダミー（MLP/Mamba版では使わない）
            level2_rotation = torch.zeros((level2_anchors.shape[0], 4), device='cuda')
            level2_rotation[:, 0] = 1.0

            # mask
            masks_level2 = torch.ones((level2_anchors.shape[0], self.n_offsets+1, 1)).float().cuda()

            # 直接Level 2を使用（voxelizeしない）
            # これにより、Level 2の数を N1 * level2_per_level1 に厳密に制御
            N2 = level2_anchors.shape[0]
            print(f"MLP/Mamba: Generated exactly {N2} Level 2 anchors from {N1} Level 1 anchors")

        # Level 2パラメータを設定
        # オプティマイザが設定されている場合は、パラメータを更新せずテンソルのデータのみ更新
        if hasattr(self, 'optimizer') and hasattr(self, '_anchor'):
            # 既存のパラメータのデータを更新
            self._anchor.data = level2_anchors
            self._anchor_feat.data = level2_feat
            self._offset.data = level2_offset
            self._scaling.data = level2_scaling

            # マスクのサイズが変わった場合のみ再作成、そうでなければ保持
            if self._mask.shape[0] != N2:
                masks_level2 = torch.ones((N2, self.n_offsets+1, 1)).float().cuda()

                # 新しいパラメータを作成（オプティマイザの状態を同期するため）
                new_mask = nn.Parameter(masks_level2.requires_grad_(True))

                # オプティマイザが既に存在する場合は、状態を更新
                if self.optimizer is not None:
                    for group in self.optimizer.param_groups:
                        if group["name"] == "mask":
                            # 古い状態を削除
                            old_param = group['params'][0]
                            if old_param in self.optimizer.state:
                                del self.optimizer.state[old_param]

                            # 新しいパラメータを設定
                            group['params'][0] = new_mask

                self._mask = new_mask
            # else: マスクの値は保持（サイズが同じなら何もしない）
        else:
            # 初期化時は新しいパラメータを作成
            self._anchor = nn.Parameter(level2_anchors.requires_grad_(True))
            self._anchor_level2 = self._anchor  # Add this line to keep a reference to level 2 anchors
            self._anchor_feat = nn.Parameter(level2_feat.requires_grad_(True))
            self._offset = nn.Parameter(level2_offset.requires_grad_(True))
            self._scaling = nn.Parameter(level2_scaling.requires_grad_(True))

            masks_level2 = torch.ones((N2, self.n_offsets+1, 1)).float().cuda()
            self._mask = nn.Parameter(masks_level2.requires_grad_(True))

    def training_setup(self, training_args):
        """
        階層的構造に対応した学習セットアップ
        """
        super().training_setup(training_args)

        # if self.use_hierarchical:
        #     # Level 1パラメータをオプティマイザに追加（マスクなし）
        #     level1_params = [
        #         {'params': [self._anchor_level1], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor_level1"},
        #         {'params': [self._offset_level1], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset_level1"},
        #         {'params': [self._anchor_feat_level1], 'lr': training_args.feature_lr, "name": "anchor_feat_level1"},
        #         {'params': [self._scaling_level1], 'lr': training_args.scaling_lr, "name": "scaling_level1"},
        #         {'params': self.mlp_level1_to_level2.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_level1_to_level2"},
        #     ]

        #     for param_group in level1_params:
        #         self.optimizer.add_param_group(param_group)
        if self.use_hierarchical:
            # モーメントマッチング版では MLP がないのでスキップ
            if self.intra_anchor_type != 'moment_matching' and self.mlp_level1_to_level2 is not None:
                # とりあえずは Level1 アンカーは固定とし、
                # Level2 と MLP だけを学習させる。
                # → optimizer には MLP だけ追加する
                self.optimizer.add_param_group({
                    'params': self.mlp_level1_to_level2.parameters(),
                    'lr': training_args.mlp_grid_lr_init,
                    "name": "mlp_level1_to_level2",
                })

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        """
        階層的モデルではLevel 1アンカーの動的調整を無効化
        Level 1は固定し、Level 2のみMLPで生成
        """
        # if not self.use_hierarchical:
            # 通常モードでは親クラスの処理を実行
        super().adjust_anchor(check_interval, success_threshold, grad_threshold, min_opacity)
        # return

        # # 階層的モードでは何もしない（Level 1アンカーを固定）
        # print("Hierarchical mode: anchor adjustment disabled (Level 1 fixed)")
        # return

        # Level 2用の統計情報とパラメータをリサイズ
        N2 = self._anchor.shape[0]

        # rotation, opacityを再初期化
        from utils.general_utils import inverse_sigmoid
        rots = torch.zeros((N2, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((N2, 1), dtype=torch.float, device="cuda"))

        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((N2), device="cuda")

        # Level 2の統計情報を再初期化
        if hasattr(self, 'opacity_accum'):
            self.opacity_accum = torch.zeros((N2, 1), device='cuda').float()
            self.anchor_demon = torch.zeros((N2, 1), device='cuda').float()
            self.offset_gradient_accum = torch.zeros((N2*self.n_offsets, 1), device='cuda').float()
            self.offset_denom = torch.zeros((N2*self.n_offsets, 1), device='cuda').float()

        print(f"Level 1 anchors: {self._anchor_level1.shape[0]}, Level 2 anchors: {N2}")

    def training_statis_level1(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        """
        Level 1アンカーの統計情報を更新
        Level 2の統計からLevel 1へ逆マッピング
        """
        if not self.use_hierarchical:
            return

        # Level 2の統計を集約してLevel 1へマッピング
        N1 = self._anchor_level1.shape[0]
        N2 = self._anchor.shape[0]
        N2_per_N1 = self.level2_per_level1

        # Level 2の実際の数がN1 * N2_per_N1と一致しない場合はスキップ
        if N2 != N1 * N2_per_N1:
            # print(f"Warning: Level 2 count mismatch (expected {N1 * N2_per_N1}, got {N2}). Skipping Level 1 statistics update.")
            return

        # Level 2のopacityをLevel 1にマッピング
        # 注意: opacityは可視的なoffsetのみを含む可能性がある
        expected_opacity_size = N2 * self.n_offsets
        actual_opacity_size = opacity.numel()

        if actual_opacity_size != expected_opacity_size:
            # opacityのサイズが一致しない場合（可視的なoffsetのみ）、統計更新をスキップ
            # これはアンカーが削除された場合やフィルタリングされた場合に発生
            return

        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])  # [N2, n_offsets]

        # Level 2 -> Level 1のマッピング（N2個のアンカー -> N1個のグループ）
        temp_opacity_grouped = temp_opacity.view(N1, N2_per_N1, self.n_offsets)  # [N1, N2_per_N1, n_offsets]
        anchor_visible_mask_grouped = anchor_visible_mask.view(N1, N2_per_N1)  # [N1, N2_per_N1]

        # Level 1の各アンカーについて、対応するLevel 2アンカーの統計を集約
        opacity_sum = (temp_opacity_grouped * anchor_visible_mask_grouped.unsqueeze(-1)).sum(dim=1).sum(dim=1, keepdim=True)  # [N1, 1]
        self.opacity_accum_level1 += opacity_sum
        self.anchor_demon_level1 += anchor_visible_mask_grouped.sum(dim=1, keepdim=True).float()

        # Level 2のgradientをLevel 1にマッピング
        anchor_visible_mask_l2 = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask_l2] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        if update_filter.sum() > 0:
            grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

            # Level 2 offset gradient -> Level 1 offset gradient
            grad_norm_expanded = torch.zeros((N1 * N2_per_N1 * self.n_offsets, 1), device='cuda')
            grad_norm_expanded[combined_mask] = grad_norm
            grad_norm_grouped = grad_norm_expanded.view(N1, N2_per_N1 * self.n_offsets, 1)  # [N1, N2_per_N1*n_offsets, 1]

            # Level 1の各offsetについて統計を集約
            for offset_idx in range(self.n_offsets):
                grad_per_offset = grad_norm_grouped[:, offset_idx::self.n_offsets, :]  # [N1, N2_per_N1, 1]
                grad_sum = grad_per_offset.sum(dim=1)  # [N1, 1]
                self.offset_gradient_accum_level1[offset_idx::self.n_offsets] += grad_sum

                mask_per_offset = combined_mask.view(N1, N2_per_N1 * self.n_offsets)[:, offset_idx::self.n_offsets]  # [N1, N2_per_N1]
                count = mask_per_offset.float().sum(dim=1, keepdim=True)  # [N1, 1]
                self.offset_denom_level1[offset_idx::self.n_offsets] += count

    def anchor_growing_level1(self, grads, threshold, offset_mask):
        """
        Level 1アンカーを成長させる
        """
        init_length = self._anchor_level1.shape[0] * self.n_offsets

        for i in range(self.update_depth):
            cur_threshold = threshold * ((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self._anchor_level1.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self._anchor_level1.unsqueeze(dim=1) + self._offset_level1 * self._scaling_level1[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size_level1 * size_factor

            grid_coords = torch.round(self._anchor_level1 / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            # 重複チェック
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for j in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[j*chunk_size:(j+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)

            from functools import reduce
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_feat = self._anchor_feat_level1.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                from torch_scatter import scatter_max
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor_level1": candidate_anchor,
                    "scaling_level1": new_scaling,
                    "anchor_feat_level1": new_feat,
                    "offset_level1": new_offsets,
                }

                # 統計情報を拡張
                temp_anchor_demon = torch.cat([self.anchor_demon_level1, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon_level1
                self.anchor_demon_level1 = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum_level1, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum_level1
                self.opacity_accum_level1 = temp_opacity_accum

                temp_offset_gradient_accum = torch.cat([self.offset_gradient_accum_level1, torch.zeros([candidate_anchor.shape[0]*self.n_offsets, 1], device='cuda').float()], dim=0)
                del self.offset_gradient_accum_level1
                self.offset_gradient_accum_level1 = temp_offset_gradient_accum

                temp_offset_denom = torch.cat([self.offset_denom_level1, torch.zeros([candidate_anchor.shape[0]*self.n_offsets, 1], device='cuda').float()], dim=0)
                del self.offset_denom_level1
                self.offset_denom_level1 = temp_offset_denom

                torch.cuda.empty_cache()

                # オプティマイザにテンソルを追加
                optimizable_tensors = self._cat_tensors_to_optimizer_level1(d)
                self._anchor_level1 = optimizable_tensors["anchor_level1"]
                self._scaling_level1 = optimizable_tensors["scaling_level1"]
                self._anchor_feat_level1 = optimizable_tensors["anchor_feat_level1"]
                self._offset_level1 = optimizable_tensors["offset_level1"]

                print(f"Added {candidate_anchor.shape[0]} new Level 1 anchors")

    def _cat_tensors_to_optimizer_level1(self, tensors_dict):
        """
        Level 1のテンソルをオプティマイザに追加
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor_level1(self, mask):
        """
        Level 1アンカーをプルーニング
        """
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer_level1(valid_points_mask)

        self._anchor_level1 = optimizable_tensors["anchor_level1"]
        self._offset_level1 = optimizable_tensors["offset_level1"]
        self._anchor_feat_level1 = optimizable_tensors["anchor_feat_level1"]
        self._scaling_level1 = optimizable_tensors["scaling_level1"]

        # 統計情報もプルーニング
        self.opacity_accum_level1 = self.opacity_accum_level1[valid_points_mask]
        self.anchor_demon_level1 = self.anchor_demon_level1[valid_points_mask]

        offset_mask = valid_points_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
        self.offset_gradient_accum_level1 = self.offset_gradient_accum_level1[offset_mask]
        self.offset_denom_level1 = self.offset_denom_level1[offset_mask]

        print(f"Pruned {mask.sum().item()} Level 1 anchors")

    def _prune_anchor_optimizer_level1(self, mask):
        """
        Level 1のオプティマイザからテンソルをプルーニング
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ["anchor_level1", "offset_level1", "anchor_feat_level1", "scaling_level1"]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                if group['name'] == "scaling_level1":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling_level1":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    @property
    def get_anchor(self):
        """
        Level 2アンカーを返す（レンダリング用）
        階層モードでは、_anchorを直接返す（初期化時に一度だけ生成済み）
        MLPの学習により、Level 2の値は間接的に更新される
        """
        if not self.use_hierarchical:
            return super(HierarchicalGaussianModel, self.__class__).get_anchor.fget(self)

        # 階層モードでは_anchorをそのまま返す（毎回再生成しない）
        return super(HierarchicalGaussianModel, self.__class__).get_anchor.fget(self)

    def conduct_encoding(self, pre_path_name):
        """
        階層構造のエンコード:
        - use_hierarchical=True のときは Level 1 + MLP だけを保存
        - decode 時に Level 2 を再生成する
        """
        if not self.use_hierarchical:
            # 通常モードは親クラスの実装をそのまま使う
            return super().conduct_encoding(pre_path_name)

        import os
        import numpy as np
        from utils.gpcc_utils import compress_gpcc
        from utils.gpcc_utils import calculate_morton_order
        import time

        print('Encoding hierarchical structure (Level 1 only)...')

        t_total_0 = time.time()

        # --- Level 1 アンカーのみをエンコード ---
        # 現状は Level 1 にマスクは無いので全点使用
        mask_anchor_level1 = torch.ones(self._anchor_level1.shape[0],
                                        dtype=torch.bool,
                                        device='cuda')

        _anchor_level1 = self._anchor_level1[mask_anchor_level1]          # [N1, 3]
        _feat_level1   = self._anchor_feat_level1[mask_anchor_level1]     # [N1, F]
        _offset_level1 = self._offset_level1[mask_anchor_level1]          # [N1, n_offsets, 3]
        _scaling_level1 = self._scaling_level1[mask_anchor_level1]        # [N1, 6]

        N1 = _anchor_level1.shape[0]

        # --- Level 1 アンカー座標を GPCC で圧縮 ---
        # 位置は voxel_size_level1 で整数グリッドにスケーリング
        _anchor_int_level1 = torch.round(_anchor_level1 / self.voxel_size_level1).int()  # [N1, 3]

        # モートンオーダで並べて空間的な局所性を上げる
        sorted_indices = calculate_morton_order(_anchor_int_level1)
        _anchor_int_level1 = _anchor_int_level1[sorted_indices]
        _feat_level1       = _feat_level1[sorted_indices]
        _offset_level1     = _offset_level1[sorted_indices]
        _scaling_level1    = _scaling_level1[sorted_indices]

        # GPCC でシリアライズして npz に保存
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc_level1.npz')
        means_strings = compress_gpcc(_anchor_int_level1)  # bytes-like
        np.savez_compressed(
            npz_path,
            voxel_size=self.voxel_size_level1,
            means_strings=np.frombuffer(means_strings, dtype=np.uint8)
        )

        bits_xyz_level1 = os.path.getsize(npz_path) * 8  # bytes→bits

        # --- その他 Level 1 パラメータと MLP を保存 ---
        torch.save(_feat_level1,   os.path.join(pre_path_name, 'anchor_feat_level1.pkl'))
        torch.save(_offset_level1, os.path.join(pre_path_name, 'offset_level1.pkl'))
        torch.save(_scaling_level1, os.path.join(pre_path_name, 'scaling_level1.pkl'))

        # Intra-Anchor MLP (Level1→Level2) のパラメータ（モーメントマッチング版では不要）
        if self.intra_anchor_type != 'moment_matching' and self.mlp_level1_to_level2 is not None:
            torch.save(self.mlp_level1_to_level2.state_dict(),
                       os.path.join(pre_path_name, 'mlp_level1_to_level2.pkl'))

        t_total = time.time() - t_total_0

        bit2MB_scale = 8 * 1024 * 1024
        log_info = (
            f"\nHierarchical Encoded (Level 1 only): "
            f"N_level1={N1}, "
            f"anchor_level1 {round(bits_xyz_level1 / bit2MB_scale, 4)} MB, "
            f"EncTime {round(t_total, 4)}s"
        )

        print(log_info)
        print(f"Expected Level 2 anchors after decoding: ~{N1 * self.level2_per_level1}")
        print(f"Compression ratio vs full Level 2: ~{self.level2_per_level1}x")

        return log_info

    def conduct_decoding(self, pre_path_name):
        """
        階層構造のデコード:
        - Level 1 を GPCC などから復元
        - MLP の重みをロード
        - Level 2 を再生成 (_generate_level2_from_level1) して使用
        """
        if not self.use_hierarchical:
            # 通常モードは親クラスの実装
            return super().conduct_decoding(pre_path_name)

        import os
        import numpy as np
        from utils.gpcc_utils import decompress_gpcc, calculate_morton_order
        from utils.general_utils import inverse_sigmoid
        import time
        import torch.nn as nn

        print('Decoding hierarchical structure...')

        t_total_0 = time.time()

        # --- Level 1 の座標を復元 ---
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc_level1.npz')
        data_dict = np.load(npz_path)

        voxel_size_level1 = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()

        # 整数グリッド座標 [N1, 3]
        _anchor_int_level1_dec = decompress_gpcc(means_strings).to('cuda')  # int32
        # モートンオーダで再度ソート（エンコード時と同じ順序に揃える）
        sorted_indices = calculate_morton_order(_anchor_int_level1_dec)
        _anchor_int_level1_dec = _anchor_int_level1_dec[sorted_indices]

        # 実座標に戻す
        anchor_level1_decoded = _anchor_int_level1_dec.float() * voxel_size_level1  # [N1, 3]

        # --- その他の Level 1 パラメータをロード ---
        anchor_feat_level1 = torch.load(os.path.join(pre_path_name, 'anchor_feat_level1.pkl')).to('cuda')
        offset_level1      = torch.load(os.path.join(pre_path_name, 'offset_level1.pkl')).to('cuda')
        scaling_level1     = torch.load(os.path.join(pre_path_name, 'scaling_level1.pkl')).to('cuda')

        # 念のためサイズチェック
        N1 = anchor_level1_decoded.shape[0]
        assert anchor_feat_level1.shape[0] == N1
        assert offset_level1.shape[0] == N1
        assert scaling_level1.shape[0] == N1

        # nn.Parameter としてセット
        self._anchor_level1 = nn.Parameter(anchor_level1_decoded, requires_grad=False)
        self._anchor_feat_level1 = nn.Parameter(anchor_feat_level1, requires_grad=False)
        self._offset_level1 = nn.Parameter(offset_level1, requires_grad=False)
        self._scaling_level1 = nn.Parameter(scaling_level1, requires_grad=False)

        # --- MLP をロード（モーメントマッチング版では不要）---
        if self.intra_anchor_type != 'moment_matching' and self.mlp_level1_to_level2 is not None:
            mlp_path = os.path.join(pre_path_name, 'mlp_level1_to_level2.pkl')
            if os.path.exists(mlp_path):
                mlp_state = torch.load(mlp_path)
                self.mlp_level1_to_level2.load_state_dict(mlp_state)

        # --- Level 2 を再生成 ---
        print('Regenerating Level 2 from Level 1...')
        self._generate_level2_from_level1()  # ここで self._anchor, _anchor_feat, _offset, _scaling, _mask が再構築される

        # rotation / opacity を初期化
        N2 = self._anchor.shape[0]
        rots = torch.zeros((N2, 4), device="cuda")
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(0.1 * torch.ones((N2, 1), dtype=torch.float, device="cuda"))

        self._rotation = nn.Parameter(rots, requires_grad=False)
        self._opacity = nn.Parameter(opacities, requires_grad=False)
        self.max_radii2D = torch.zeros((N2,), device='cuda')

        self.decoded_version = True

        t_total = time.time() - t_total_0
        log_info = (
            f"\nHierarchical Decoded: "
            f"Level1={self._anchor_level1.shape[0]}, "
            f"Level2={self._anchor.shape[0]}, "
            f"DecTime {round(t_total, 4)}s"
        )

        print(log_info)

        return log_info

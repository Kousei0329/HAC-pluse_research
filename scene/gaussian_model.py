#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order

from .masked_conv import MaskedConv2d, MaskedConv1d
from .pointnet import PointNet

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i

class SpatialContextModule(nn.Module):
    """
    近傍アンカーの空間情報を集約してコンテキストとして利用するモジュール
    """
    def __init__(self, k_neighbors=16, feat_dim=48):
        super().__init__()
        self.k = k_neighbors
        # 近傍情報の集約
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),  # hash_feat + relative_pos
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        # アテンションベースの重み付け
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1)
        )

    def forward(self, anchors, hash_feats):
        """
        Args:
            anchors: [N, 3] アンカー座標
            hash_feats: [N, feat_dim] ハッシュ特徴
        Returns:
            [N, feat_dim] 空間コンテキストが統合されたハッシュ特徴
        """
        N = anchors.shape[0]

        # 少数のアンカーの場合はスキップ
        if N < self.k:
            return hash_feats

        # K-NN探索（メモリ効率的な実装）
        # バッチサイズを制限して処理
        batch_size = 1024  # メモリに応じて調整可能
        k = min(self.k + 1, N)  # 自分自身を含むため+1

        all_indices = []

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_anchors = anchors[i:end_i]  # [B, 3]

            # バッチごとに距離計算
            # [B, N] の距離行列（全体ではなくバッチのみ）
            dists = torch.cdist(batch_anchors, anchors)  # [B, N]

            # 各点のK近傍を取得
            _, batch_indices = torch.topk(dists, k=k, largest=False, dim=-1)  # [B, K]
            all_indices.append(batch_indices)

        # 全バッチの結果を結合
        indices = torch.cat(all_indices, dim=0)  # [N, K]

        # 自分自身を除外（最も近い点は自分自身のため）
        indices = indices[:, 1:]  # [N, K-1]

        # 近傍特徴の取得
        neighbor_feats = hash_feats[indices]  # [N, K-1, feat_dim]
        relative_pos = anchors.unsqueeze(1) - anchors[indices]  # [N, K-1, 3]

        # 特徴変換
        combined = torch.cat([neighbor_feats, relative_pos], dim=-1)
        neighbor_context = self.neighbor_mlp(combined)  # [N, K-1, feat_dim]

        # アテンション重み
        attn_weights = torch.softmax(
            self.attention(neighbor_context), dim=1
        )  # [N, K-1, 1]

        # コンテキスト集約
        spatial_context = (neighbor_context * attn_weights).sum(dim=1)  # [N, feat_dim]

        # 元のハッシュ特徴と融合（残差接続）
        return hash_feats + spatial_context

class JointContextModule(nn.Module):
	def __init__(self, dim_in):
		super(JointContextModule, self).__init__()
		# Use Conv1d for coordinate data [N, 3]
		# Input: coordinates x [N, 3]
		# Output: context features [N, 384]
		self.masked = MaskedConv1d("A", in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=2)

		# MLP to combine hash features and context features
		# Input: hash_feats (dim_in) + context (384) = dim_in + 384
		# Output: dim_in (same as hash features)
		self.fusion_mlp = nn.Sequential(
			nn.Linear(dim_in + 48, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, dim_in)
		)

	def forward(self, x, hash_feats):
		# Apply masked conv to coordinates to get context features
		# x: [N, 3] coordinates
		context_feats = self.masked(x)  # [N, 384]
		# Concatenate hash features and context features
		combined = torch.cat([hash_feats, context_feats], dim=1)  # [N, dim_in + 384]
		# Pass through MLP to get final features with same dimension as hash_feats
		output = self.fusion_mlp(combined)  # [N, dim_in]
		return output

class JointPointNetModule(nn.Module):
    def __init__(self, dim_in):
        super(JointPointNetModule, self).__init__()
        self.pointnet = PointNet(out_dim=48)

        # MLP to combine hash features and context features
        # Input: hash_feats (dim_in) + context (48) = dim_in + 48
        # Output: dim_in (same as hash features)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim_in + 48, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_in)
        )

    def forward(self, x, hash_feats):
        # Apply PointNet to coordinates to get context features
        # x: [N, 3] coordinates
        context_feats = self.pointnet(x)  # [N, 48]
        # Concatenate hash features and context features
        combined = torch.cat([hash_feats, context_feats], dim=1)  # [N, dim_in + 48]
        # Pass through MLP to get final features with same dimension as hash_feats
        output = self.fusion_mlp(combined)  # [N, dim_in]
        return output


class Channel_CTX_fea(nn.Module):
    def __init__(self, use_gated=False):
        super().__init__()
        self.use_gated = use_gated

        if use_gated:
            # Gated MLP (10分割): 出力に gate を追加 (mean, scale, prob, gate の4つ)
            # 各チャネル5次元
            self.MLP_d0 = nn.Sequential(
                nn.Linear(50*3+5*0, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),  # 5*4 = mean(5) + scale(5) + prob(5) + gate(5)
            )
            self.MLP_d1 = nn.Sequential(
                nn.Linear(50*3+5*1, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d2 = nn.Sequential(
                nn.Linear(50*3+5*2, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d3 = nn.Sequential(
                nn.Linear(50*3+5*3, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d4 = nn.Sequential(
                nn.Linear(50*3+5*4, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d5 = nn.Sequential(
                nn.Linear(50*3+5*5, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d6 = nn.Sequential(
                nn.Linear(50*3+5*6, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d7 = nn.Sequential(
                nn.Linear(50*3+5*7, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d8 = nn.Sequential(
                nn.Linear(50*3+5*8, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*4),
            )
            self.MLP_d9 = nn.Sequential(
                nn.Linear(50*3+5*9, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 5*3),  # 最後のチャネルはgateを出力しない
            )
        else:
            # 元の実装 (5分割)
            self.MLP_d0 = nn.Sequential(
                nn.Linear(50*3+10*0, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 10*3),
            )
            self.MLP_d1 = nn.Sequential(
                nn.Linear(50*3+10*1, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 10*3),
            )
            self.MLP_d2 = nn.Sequential(
                nn.Linear(50*3+10*2, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 10*3),
            )
            self.MLP_d3 = nn.Sequential(
                nn.Linear(50*3+10*3, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 10*3),
            )
            self.MLP_d4 = nn.Sequential(
                nn.Linear(50*3+10*4, 20*2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(20*2, 10*3),
            )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]

        if self.use_gated:
            # 10分割: 各5次元
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = torch.split(fea_q, split_size_or_sections=[5]*10, dim=-1)

            # Gated MLP: gate で情報を選択的に伝播
            out_d0 = self.MLP_d0(torch.cat([mean_scale], dim=-1))
            mean_d0, scale_d0, prob_d0, gate_d0 = torch.chunk(out_d0, chunks=4, dim=-1)
            gate_d0 = torch.sigmoid(gate_d0)

            out_d1 = self.MLP_d1(torch.cat([d0 * gate_d0, mean_scale], dim=-1))
            mean_d1, scale_d1, prob_d1, gate_d1 = torch.chunk(out_d1, chunks=4, dim=-1)
            gate_d1 = torch.sigmoid(gate_d1)

            out_d2 = self.MLP_d2(torch.cat([d0 * gate_d0, d1 * gate_d1, mean_scale], dim=-1))
            mean_d2, scale_d2, prob_d2, gate_d2 = torch.chunk(out_d2, chunks=4, dim=-1)
            gate_d2 = torch.sigmoid(gate_d2)

            out_d3 = self.MLP_d3(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, mean_scale], dim=-1))
            mean_d3, scale_d3, prob_d3, gate_d3 = torch.chunk(out_d3, chunks=4, dim=-1)
            gate_d3 = torch.sigmoid(gate_d3)

            out_d4 = self.MLP_d4(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, mean_scale], dim=-1))
            mean_d4, scale_d4, prob_d4, gate_d4 = torch.chunk(out_d4, chunks=4, dim=-1)
            gate_d4 = torch.sigmoid(gate_d4)

            out_d5 = self.MLP_d5(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, d4 * gate_d4, mean_scale], dim=-1))
            mean_d5, scale_d5, prob_d5, gate_d5 = torch.chunk(out_d5, chunks=4, dim=-1)
            gate_d5 = torch.sigmoid(gate_d5)

            out_d6 = self.MLP_d6(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, d4 * gate_d4, d5 * gate_d5, mean_scale], dim=-1))
            mean_d6, scale_d6, prob_d6, gate_d6 = torch.chunk(out_d6, chunks=4, dim=-1)
            gate_d6 = torch.sigmoid(gate_d6)

            out_d7 = self.MLP_d7(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, d4 * gate_d4, d5 * gate_d5, d6 * gate_d6, mean_scale], dim=-1))
            mean_d7, scale_d7, prob_d7, gate_d7 = torch.chunk(out_d7, chunks=4, dim=-1)
            gate_d7 = torch.sigmoid(gate_d7)

            out_d8 = self.MLP_d8(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, d4 * gate_d4, d5 * gate_d5, d6 * gate_d6, d7 * gate_d7, mean_scale], dim=-1))
            mean_d8, scale_d8, prob_d8, gate_d8 = torch.chunk(out_d8, chunks=4, dim=-1)
            gate_d8 = torch.sigmoid(gate_d8)

            out_d9 = self.MLP_d9(torch.cat([d0 * gate_d0, d1 * gate_d1, d2 * gate_d2, d3 * gate_d3, d4 * gate_d4, d5 * gate_d5, d6 * gate_d6, d7 * gate_d7, d8 * gate_d8, mean_scale], dim=-1))
            mean_d9, scale_d9, prob_d9 = torch.chunk(out_d9, chunks=3, dim=-1)

            mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4, mean_d5, mean_d6, mean_d7, mean_d8, mean_d9], dim=-1)
            scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4, scale_d5, scale_d6, scale_d7, scale_d8, scale_d9], dim=-1)
            prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4, prob_d5, prob_d6, prob_d7, prob_d8, prob_d9], dim=-1)

            if to_dec == 0:
                return mean_d0, scale_d0, prob_d0
            if to_dec == 1:
                return mean_d1, scale_d1, prob_d1
            if to_dec == 2:
                return mean_d2, scale_d2, prob_d2
            if to_dec == 3:
                return mean_d3, scale_d3, prob_d3
            if to_dec == 4:
                return mean_d4, scale_d4, prob_d4
            if to_dec == 5:
                return mean_d5, scale_d5, prob_d5
            if to_dec == 6:
                return mean_d6, scale_d6, prob_d6
            if to_dec == 7:
                return mean_d7, scale_d7, prob_d7
            if to_dec == 8:
                return mean_d8, scale_d8, prob_d8
            if to_dec == 9:
                return mean_d9, scale_d9, prob_d9
            return mean_adj, scale_adj, prob_adj
        else:
            # 元の実装 (5分割)
            d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
            mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)

            mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
            scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
            prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

            if to_dec == 0:
                return mean_d0, scale_d0, prob_d0
            if to_dec == 1:
                return mean_d1, scale_d1, prob_d1
            if to_dec == 2:
                return mean_d2, scale_d2, prob_d2
            if to_dec == 3:
                return mean_d3, scale_d3, prob_d3
            if to_dec == 4:
                return mean_d4, scale_d4, prob_d4
            return mean_adj, scale_adj, prob_adj

import torch
import torch.nn as nn
import torch.nn.functional as F


# class Channel_CTX_fea(nn.Module):
#     """
#     Transformer版 Channel Context モジュール

#     fea_q: [N, 50]  -> 10トークン (各5次元)
#     mean_scale: [N, 150] (たぶん 50*3)

#     use_gated:
#       - False: 各トークンごとに mean(5), scale(5), prob(5) の 15次元
#       - True : 各トークンごとに mean(5), scale(5), prob(5), gate(5) の 20次元
#                （最後の to_dec=9 ケースでも gate まで出しておく）

#     Transformer で 10チャネル分を一気に処理して、
#     最後に [N, 10, 5] -> [N, 50] に reshape して返す。
#     """

#     def __init__(
#         self,
#         use_gated: bool = False,
#         d_model: int = 64,
#         nhead: int = 4,
#         num_layers: int = 2,
#         dim_feedforward: int = 256,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.use_gated = use_gated
#         self.d_model = d_model

#         # fea_q を 5次元 → d_model に写像
#         self.fea_embed = nn.Linear(5, d_model)

#         # mean_scale (たぶん 150次元) を d_model に圧縮して、全トークンに足し込む
#         self.cond_proj = nn.Linear(50 * 3, d_model)

#         # 位置埋め込み (10トークン固定想定)
#         self.pos_embed = nn.Parameter(torch.zeros(1, 10, d_model))
#         nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,  # [B, L, D] 形式を使う
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # 出力ヘッド
#         if use_gated:
#             out_dim = 5 * 4  # mean, scale, prob, gate
#         else:
#             out_dim = 5 * 3  # mean, scale, prob

#         self.out_proj = nn.Linear(d_model, out_dim)

#     def forward(self, fea_q: torch.Tensor, mean_scale: torch.Tensor, to_dec: int = -1):
#         """
#         fea_q: [N, 50]
#         mean_scale: [N, 150] (想定)
#         to_dec:
#           - -1 のとき: 全チャネル分 [N, 50] を返す
#           - 0〜9 のとき: 対応するチャネルだけ [N, 5] を返す
#         """
#         B, C = fea_q.shape
#         assert C == 50, f"fea_q のチャネル数が想定(50)と違う: {C}"

#         # 10トークン × 5次元に分割: [N, 50] -> [N, 10, 5]
#         fea_seq = fea_q.view(B, 10, 5)

#         # 入力埋め込み
#         x = self.fea_embed(fea_seq)  # [N, 10, d_model]

#         # コンディション (mean_scale) を全トークンにブロードキャストして加算
#         assert mean_scale.dim() == 2, "mean_scale は [N, 150] の2次元テンソルを想定"
#         cond = self.cond_proj(mean_scale)  # [N, d_model]
#         cond = cond.unsqueeze(1)  # [N, 1, d_model]
#         x = x + cond + self.pos_embed  # [N, 10, d_model]

#         # Transformer Encoder
#         # ※ ここでは因果マスク無し（全チャネル双方向注意）にしている。
#         #   ARな条件付き分布を厳密に守りたければ causal mask を入れる必要あり。
#         x = self.encoder(x)  # [N, 10, d_model]

#         # 出力: [N, 10, out_dim]
#         out = self.out_proj(x)

#         if self.use_gated:
#             # [N, 10, 20] -> 各 5次元に分割
#             mean, scale, prob, gate = torch.chunk(out, chunks=4, dim=-1)
#             gate = torch.sigmoid(gate)

#             # to_dec 指定がある場合は、そのチャネルのみを返す
#             if 0 <= to_dec <= 9:
#                 # [N, 5] を返す
#                 return (
#                     mean[:, to_dec, :],
#                     scale[:, to_dec, :],
#                     prob[:, to_dec, :],
#                     gate[:, to_dec, :],
#                 )

#             # 全チャネルまとめて [N, 50] で返す
#             mean_adj = mean.reshape(B, -1)   # [N, 50]
#             scale_adj = scale.reshape(B, -1) # [N, 50]
#             prob_adj = prob.reshape(B, -1)   # [N, 50]

#             return mean_adj, scale_adj, prob_adj, gate.reshape(B, -1)

#         else:
#             # [N, 10, 15] -> 各 5次元に分割
#             mean, scale, prob = torch.chunk(out, chunks=3, dim=-1)

#             if 0 <= to_dec <= 9:
#                 return (
#                     mean[:, to_dec, :],
#                     scale[:, to_dec, :],
#                     prob[:, to_dec, :],
#                 )

#             mean_adj = mean.reshape(B, -1)   # [N, 50]
#             scale_adj = scale.reshape(B, -1) # [N, 50]
#             prob_adj = prob.reshape(B, -1)   # [N, 50]

#             return mean_adj, scale_adj, prob_adj



class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

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
                 use_joint_context: bool=False,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.use_spatial_context = use_spatial_context
        self.use_joint_context = use_joint_context

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        # ここでガウス分布パラメータを予測
        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+feat_dim+1+1+1),
        ).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea(use_gated=use_gated_mlp).cuda()
            if use_gated_mlp:
                print('Using Gated MLP for Channel Context')
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()

        # 空間コンテキストモジュールの初期化
        if self.use_spatial_context:
            self.spatial_context_module = SpatialContextModule(
                k_neighbors=16,
                feat_dim=self.encoding_xyz.output_dim
            ).cuda()
            print('Using Spatial Context Module for hash features')

        # ジョイントコンテキストモジュールの初期化
        print("ジョイントコンテキスト起動してくれ")
        if self.use_joint_context:
            # # Use the actual output dimension from encoding_xyz
            # encoding_output_dim = self.encoding_xyz.output_dim
            # self.joint_context_module = JointContextModule(encoding_output_dim).cuda()
            # print(f'Using Joint Context Module for joint features (dim={encoding_output_dim})')
            # Use the actual output dimension from encoding_xyz
            encoding_output_dim = self.encoding_xyz.output_dim
            self.joint_context_module = JointPointNetModule(encoding_output_dim).cuda()
            print(f'PointNetモジュール使用中(dim={encoding_output_dim})')
        print("ジョイントコンテキスト起動してくれた?")
    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._mask,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask[:, :10, :]
        mask_sig = torch.sigmoid(self._mask[:, :10, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        mask = self.get_mask  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0

        # 正規化前のアンカー座標を保存（空間コンテキスト用）
        x_orig = x.clone()  # if self.use_spatial_context else None

        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]

        # 空間コンテキストモジュールを適用
        if self.use_spatial_context:
            features = self.spatial_context_module(x_orig, features)
        
        # Jointモデルの自己回帰モジュール適用
        # アンカー座標に対してContextPredictionを適用するイメージ 
        '''
        class ContextPrediction(nn.Module):
            def __init__(self, dim_in):
                super(ContextPrediction, self).__init__()
                self.masked = MaskedConv2d("A", in_channels=dim_in, out_channels=384, kernel_size=5, stride=1, padding=2)
            
            def forward(self, x):
                return self.masked(x)
        '''
        if self.use_joint_context:
            features = self.joint_context_module(x_orig, features)
        
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets+1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        # マスクの学習率を決定（freeze_maskがTrueの場合は0に設定）
        mask_lr = 0.0 if training_args.freeze_mask else training_args.mask_lr_init * self.spatial_lr_scale
        if training_args.freeze_mask:
            print("Mask learning is FROZEN (learning rate = 0.0)")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': mask_lr, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': mask_lr, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        # マスクスケジューラー（freeze_maskがTrueの場合は常に0を返す）
        if training_args.freeze_mask:
            self.mask_scheduler_args = lambda x: 0.0
        else:
            self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.mask_lr_delay_mult,
                                                        max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
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

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets+1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    def build_level2_from_level1(self, voxel_size_L2: float):
        """
        Level1アンカー群からLevel2アンカーを生成する（モーメントマッチング）

        Level1の各アンカーを3Dガウシアンとみなし、voxel_size_L2でグループ化。
        各グループをガウシアン混合として扱い、モーメントマッチングで
        1つのガウシアンに近似してLevel2アンカーを生成する。

        Args:
            voxel_size_L2 (float): Level2のボクセルサイズ

        Returns:
            tuple: (anchor_L2, scaling_L2, rotation_L2, feat_L2, offset_L2, mask_L2)
                anchor_L2:   [N2, 3]      Level2アンカー位置
                scaling_L2:  [N2, 6]      Level2スケーリング（log space）
                rotation_L2: [N2, 4]      Level2回転（quaternion）
                feat_L2:     [N2, feat_dim] Level2特徴
                offset_L2:   [N2, n_offsets, 3] Level2オフセット
                mask_L2:     [N2, n_offsets+1, 1] Level2マスク
        """
        device = self.get_anchor.device

        # Level1のアンカー情報を取得
        anchors_L1 = self.get_anchor  # [N1, 3]
        scaling_L1 = self.get_scaling  # [N1, 6]
        rotation_L1 = self.get_rotation  # [N1, 4]
        feat_L1 = self._anchor_feat  # [N1, feat_dim]

        N1 = anchors_L1.shape[0]

        if N1 == 0:
            # 空の場合は空のテンソルを返す
            empty_anchor = torch.empty(0, 3, device=device)
            empty_scaling = torch.empty(0, 6, device=device)
            empty_rotation = torch.empty(0, 4, device=device)
            empty_feat = torch.empty(0, self.feat_dim, device=device)
            empty_offset = torch.empty(0, self.n_offsets, 3, device=device)
            empty_mask = torch.empty(0, self.n_offsets + 1, 1, device=device)
            return empty_anchor, empty_scaling, empty_rotation, empty_feat, empty_offset, empty_mask

        # 1. Level1 → Level2 のグルーピング
        # ボクセルグリッド座標を計算
        grid_L2 = torch.round(anchors_L1 / voxel_size_L2).long()  # [N1, 3]

        # ユニークなセルとインデックスを取得
        # unique は連続したメモリを要求するため、contiguous() を呼ぶ
        grid_L2_flat = grid_L2[:, 0] * 1000000 + grid_L2[:, 1] * 1000 + grid_L2[:, 2]  # [N1]
        unique_cells, inverse_idx = torch.unique(grid_L2_flat, return_inverse=True)  # unique_cells: [N2], inverse_idx: [N1]

        N2 = unique_cells.shape[0]

        # 2. 各グループcについてモーメントマッチング
        anchor_L2_list = []
        scaling_L2_list = []
        rotation_L2_list = []
        feat_L2_list = []

        for c in range(N2):
            # グループcに属するLevel1アンカーのマスク
            idx_c = (inverse_idx == c)  # [N1] bool

            # グループ内のアンカー
            a_i = anchors_L1[idx_c]  # [M, 3]
            sc_i = scaling_L1[idx_c]  # [M, 6]
            rt_i = rotation_L1[idx_c]  # [M, 4]
            f_i = feat_L1[idx_c]  # [M, feat_dim]

            M = a_i.shape[0]

            # 重み（全て1）
            w = torch.ones(M, device=device)  # [M]
            w_sum = w.sum()

            # 新しい平均（Level2アンカー位置）
            mu_L2 = (w[:, None] * a_i).sum(dim=0) / w_sum  # [3]

            # 各Level1アンカーの共分散行列を計算
            # self.covariance_activation は build_covariance_from_scaling_rotation
            # scaling: [M, 6], rotation: [M, 4] -> covariance: [M, 3, 3] (対称行列の6要素表現)
            # strip_symmetric で [M, 6] になっている可能性があるので、完全な [M, 3, 3] に戻す

            # build_scaling_rotation を使って L @ L^T を計算
            from utils.general_utils import build_scaling_rotation

            # scaling_activation で exp を適用（既に get_scaling で適用済み）
            # sc_i は既に exp 済み
            L_i = build_scaling_rotation(sc_i, rt_i)  # [M, 3, 3]
            Sigma_i = L_i @ L_i.transpose(1, 2)  # [M, 3, 3]

            # 新しい共分散（Level2共分散）
            # Σ_L2 = Σ_i w_i (Σ_i + (μ_i - μ_L2)(μ_i - μ_L2)^T) / Σ_i w_i
            diff = a_i - mu_L2.unsqueeze(0)  # [M, 3]
            outer = diff.unsqueeze(2) @ diff.unsqueeze(1)  # [M, 3, 3]

            Sigma_L2 = (w[:, None, None] * (Sigma_i + outer)).sum(dim=0) / w_sum  # [3, 3]

            # 3. 共分散 → (scaling, rotation) への変換
            # 固有値分解
            eigvals, eigvecs = torch.linalg.eigh(Sigma_L2)  # eigvals: [3], eigvecs: [3, 3]

            # 固有値をクランプ（数値安定性）
            eigvals = torch.clamp(eigvals, min=1e-6)

            # スケール = sqrt(固有値)
            scales = torch.sqrt(eigvals)  # [3]

            # 回転行列 → クォータニオン
            R = eigvecs  # [3, 3]
            quat = self._rotation_matrix_to_quaternion(R)  # [4]

            # scaling ベクトルは 6次元
            # 最初の3次元: log(scales), 残り3次元: 0
            log_scales = torch.log(scales)  # [3]
            scaling_vec = torch.cat([log_scales, torch.zeros(3, device=device)], dim=0)  # [6]

            # Level2の特徴：重み付き平均
            feat_L2 = (w[:, None] * f_i).sum(dim=0) / w_sum  # [feat_dim]

            # リストに追加
            anchor_L2_list.append(mu_L2)
            scaling_L2_list.append(scaling_vec)
            rotation_L2_list.append(quat)
            feat_L2_list.append(feat_L2)

        # スタック
        anchor_L2 = torch.stack(anchor_L2_list, dim=0)  # [N2, 3]
        scaling_L2 = torch.stack(scaling_L2_list, dim=0)  # [N2, 6]
        rotation_L2 = torch.stack(rotation_L2_list, dim=0)  # [N2, 4]
        feat_L2 = torch.stack(feat_L2_list, dim=0)  # [N2, feat_dim]

        # 4. Level2の offset / mask の初期化
        # offset: 全てゼロ
        offset_L2 = torch.zeros((N2, self.n_offsets, 3), device=device)  # [N2, n_offsets, 3]

        # mask: 全て1
        mask_L2 = torch.ones((N2, self.n_offsets + 1, 1), device=device)  # [N2, n_offsets+1, 1]

        return anchor_L2, scaling_L2, rotation_L2, feat_L2, offset_L2, mask_L2

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        3x3回転行列をクォータニオン [w, x, y, z] に変換

        Args:
            R: [3, 3] 回転行列

        Returns:
            [4] クォータニオン（正規化済み）
        """
        # Shepperdのアルゴリズムを使用
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        quat = torch.stack([w, x, y, z], dim=0)

        # 正規化
        quat = quat / torch.norm(quat)

        return quat

    @torch.no_grad()
    def compute_total_bits(self):
        """
        全anchorを使って正確なビット数を計算する
        Returns:
            (bit_feat, bit_scaling, bit_offsets): 各パラメータの総ビット数
        """
        # AQM設定用のパラメータ
        Q_feat = 1
        Q_scaling = 0.1
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]

        feat_context = self.calc_interp_feat(_anchor)
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)

        # conduct_encoding()と同じようにチャネルごとのQを使う
        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])  # [N, 50]
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)  # [N*6]
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)  # [N*30]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))  # [N, 50]
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))  # [N*6]
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N*30]
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        probs = torch.stack([prob, prob_adj], dim=-1)
        probs = torch.softmax(probs, dim=-1)

        # conduct_encoding()と同じ形状にする
        mean_scaling = mean_scaling.contiguous().view(-1)
        scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        mean_offsets = mean_offsets.contiguous().view(-1)
        scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        scale = torch.clamp(scale, min=1e-9)

        grid_scaling = (STE_multistep.apply(_scaling.view(-1), Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets.view(-1, 3*self.n_offsets).view(-1), Q_offsets)).detach()
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1)

        bit_feat = self.EG_mix_prob_2.forward(_feat,
                                            mean, mean_adj,
                                            scale, scale_adj,
                                            probs[..., 0], probs[..., 1],
                                            Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_feat_total = torch.sum(bit_feat).item()
        bit_scaling_total = torch.sum(bit_scaling).item()
        bit_offsets_total = torch.sum(bit_offsets).item()

        return bit_feat_total, bit_scaling_total, bit_offsets_total

    @torch.no_grad()
    def estimate_final_bits(self):

        # AQM設定用のパラメータ
        # 明らかにスケーリングのみ値が高い
        Q_feat = 1
        # Q_scaling = 0.001
        Q_scaling = 0.1
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]

        # conduct_encoding()と同じようにチャネルごとのQを使う
        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])  # [N, 50]
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)  # [N*6]
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)  # [N*30]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))  # [N, 50]
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))  # [N*6]
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N*30]
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        # mean_adj, scale_adj, prob_adj, gate = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        probs = torch.stack([prob, prob_adj], dim=-1)
        probs = torch.softmax(probs, dim=-1)

        # conduct_encoding()と同じ形状にする
        mean_scaling = mean_scaling.contiguous().view(-1)
        scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        mean_offsets = mean_offsets.contiguous().view(-1)
        scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        scale = torch.clamp(scale, min=1e-9)

        grid_scaling = (STE_multistep.apply(_scaling.view(-1), Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets.view(-1, 3*self.n_offsets).view(-1), Q_offsets)).detach()
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1)

        bit_feat = self.EG_mix_prob_2.forward(_feat,
                                            mean, mean_adj,
                                            scale, scale_adj,
                                            probs[..., 0], probs[..., 1],
                                            Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0
        t_codec = 0

        t_total_0 = get_time()
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]  # N, 50
        _grid_offsets = self._offset[mask_anchor]  # N, 10, 3
        _scaling = self.get_scaling[mask_anchor]  # N, 6
        _mask = self.get_mask[mask_anchor]  # N, 10, 1

        N = _anchor.shape[0]

        t_anchor_0 = get_time()
        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path= os.path.join(pre_path_name, 'xyz_gpcc.npz')
        means_strings = compress_gpcc(_anchor_int)
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
        bits_xyz = os.path.getsize(npz_path) * 8
        t_anchor += get_time() - t_anchor_0

        _anchor = _anchor_int * self.voxel_size
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask = _mask[sorted_indices]

        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        # データをbatch_sizeごとに分割し、処理
        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            # AQM用の初期パラメータ
            Q_feat = 1
            # Q_scaling = 0.001
            Q_scaling = 0.1
            Q_offsets = 0.2

            # 現在のステップに応じたアンカーを切り出し
            anchor_slice = _anchor[N_start:N_end]

            # encode feat
            # features = self.spatial_context_module(x_orig, features)を適用
            # ここでハッシュエンコーディングも適用される
            '''
            calc_interp_feat(anchor_slice)の流れ
            1. 元座標を保存
            2. 座標の正規化
            3. ハッシュエンコーディング(GridEncoderまたは,mix_3D2D_encoding)を適用し、3D座標から特徴量を取得
                3.1 空間コンテキストをspatial_context_moduleで計算し特徴量に統合
            4. 取得した特徴量を返す
            

            mix_3D2D_encoding(
                n_features,
                resolution_list,
                log2_hashmap_size,
                resolution_list_2D,
                log2_hashmap_size_2D,
                ste_binary,
                ste_multiscale,
                add_noise,
                Q,
            )の流れ
            前提として、GridEncordingを複数使用している
            2Dを使用する場合、2D平面に投影して取得される特徴量も獲得する
            xyz, xy, yz, zxの4種類のGridEncodingを使用し、各特徴量を結合して最終的な特徴量を生成する
            xyzは2*16、xy,yz,zxは2*8の特徴量を生成する

            GridEncorder(
                num_dim,(2D or 3D)
                n_features, (各レベルの特徴次元数)
                resolution_list, (ハッシュグリッド解像度リスト)
                log2_hashmap_size, (最大ハッシュテーブルサイズのlog2)
            )
            グリッドエンコーディングとは、連続座標を高次元の特徴ベクトルe(x)に移す写像

            1. ハッシュテーブルを構築
                低解像度レベル: 全グリッド点を保存
                高解像度レベル: ハッシュ衝突を許容（メモリ節約）
                例（3Dの場合）:
                    レベル	解像度	理想数 (res³)	実際数 (min)	オフセット(全レベルの合計パラメータ)
                    0	18	5,832	5,832	0
                    1	24	13,824	13,824	5,832
                    2	33	35,937	35,944	19,656
                    ...	...	...	...	...
                    9	275	20,796,875	524,288	...
                    10	376	53,157,376	524,288	...
                    11	514	135,796,744	524,288	...
            2. 


            '''

            feat_context = self.calc_interp_feat(anchor_slice)  # [N_num, ?]

            # many [N_num, ?]
            # 1つのMLPで全てのガウシアンパラメータの分布（平均・スケール）と量子化パラメータを一度に予測
            # ここがHash Assisted Context
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            # 調整パラメータの次元合わせ
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            
            # 統計パラメータの整形(スケーリングとオフセットの平均、分散を取得)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            
            # 適応的な調整の適用
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            # 特徴量の抽出
            feat = _feat[N_start:N_end]
            # 特徴量をSTEで量子化
                # STEとはStraight-Through Estimatorの略で、量子化などの非微分可能な操作を微分可能に近似する手法
            feat = STE_multistep.apply(feat, Q_feat, self._anchor_feat.mean())
            torch.cuda.synchronize(); t0 = time.time()

            t_feature_0 = get_time()
            # 特徴量の平均、分散、確率・重みを結合
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = torch.clamp(scale, min=1e-9)
            bit_feat = 0

            # use_gated_mlp に応じて分割数とチャネルサイズを変更
            if self.mlp_deform.use_gated:
                num_chunks = 10  # 10分割
                chunk_size = 5   # 各5次元
            else:
                num_chunks = 5   # 5分割
                chunk_size = 10  # 各10次元

            # チャンクごとにループ
            # ここからIntra-Anchor Context
            for cc in range(num_chunks):
                # mean_adj, scale_adj, prob_adj, gate = self.get_deform_mlp.forward(feat, mean_scale, to_dec=cc)
                # 調整パラメータを予測
                mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat, mean_scale, to_dec=cc)
                # HAC予測の確率とIntra-Anchorの確率を結合
                # 重みの計算
                probs = torch.stack([prob[:, cc*chunk_size:cc*chunk_size+chunk_size], prob_adj], dim=-1)
                probs = torch.softmax(probs, dim=-1)

                # チャンクごとの特徴量と量子化パラメータを取得
                feat_tmp = feat[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1)
                Q_feat_tmp = Q_feat[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1)

                # Gaussian Mixtureを実行
                '''
                encoder_gaussian_mixed_chunk(
                    x, 
                    mean_list,  
                    scale_list, 
                    probs_list, 
                    Q, 
                    file_name, 
                    chunk_size
                )
                の流れ
                1. xをQを使用して量子化
                2. ガウス分布N(mean, scale)とN(mean_adj, scale_adj)のCDF(累積分布関数)を計算
                3. xを正規化
                4. xとCDFを使用し算術符号化(AE)を実行
                5. file_nameにビット列を保存
                6. ビット数を返す
                '''
                bit_feat += encoder_gaussian_mixed_chunk(
                    feat_tmp,
                    [mean[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                    [scale[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_feat_tmp,
                    file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
            t_feature += get_time() - t_feature_0

            # GPU同期と時間計測、ビット数取得
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)


            t_scaling_0 = get_time()
            # スケーリングの抽出と量子化
            scaling = _scaling[N_start:N_end].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            # スケーリングを単一ガウス分布でエンコード
            '''
            encoder_gaussian_chunk(x, mean, scale, Q, file_name, chunk_size)の流れ
            1. xとQを使用し量子化
            2. ガウス分布N(mean, scale)のCDF(累積分布関数)を計算
            3. xを正規化
            4. xとCDFを使用し算術符号化(AE)を実行
            5. file_nameにビット列を保存
            6. ビット数を返す
            '''
            bit_scaling = encoder_gaussian_chunk(
                scaling, 
                mean_scaling, 
                scale_scaling, 
                Q_scaling, 
                file_name=scaling_b_name, 
                chunk_size=10_0000
            )
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            mask = _mask[N_start:N_end]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
            # オフセットの抽出と量子化
            offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets = encoder_gaussian_chunk(
                offsets[mask], 
                mean_offsets[mask], 
                scale_offsets[mask], 
                Q_offsets[mask], 
                file_name=offsets_b_name, 
                chunk_size=10_0000
            )
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            t_offset += get_time() - t_offset_0

            torch.cuda.empty_cache()

        bit_anchor = bits_xyz
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        t_hash_0 = get_time()
        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bit_hash = encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel()*32
        t_hash += get_time() - t_hash_0

        t_mask_0 = get_time()
        bit_masks = encoder(_mask, file_name=masks_b_name)
        t_mask += get_time() - t_mask_0

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        # 32*3*2/bit2MB_scale is for xyz_bound_min and xyz_bound_max
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale + 32*3*2/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        log_info_time = f"\nEncoded time in s: " \
                   f"anchor {round(t_anchor, 4)}, " \
                   f"feat {round(t_feature, 4)}, " \
                   f"scaling {round(t_scaling, 4)}, " \
                   f"offsets {round(t_offset, 4)}, " \
                   f"hash {round(t_hash, 4)}, " \
                   f"masks {round(t_mask, 4)}, " \
                   f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time
        return log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0

        t_total_0 = get_time()

        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')

        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))

        xyz_decoded_list = []
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        t_anchor_0 = get_time()
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        data_dict = np.load(npz_path)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * voxel_size
        t_anchor += get_time() - t_anchor_0
        N = anchor_decoded.shape[0]

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        t_mask_0 = get_time()
        masks_decoded = decoder(N*self.n_offsets, masks_b_name)  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)
        t_mask += get_time() - t_mask_0

        t_hash_0 = get_time()
        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder(N_hash, hash_b_name)  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)
        t_hash += get_time() - t_hash_0

        for s in range(steps):

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            # Q_scaling = 0.001
            Q_scaling = 0.1
            Q_offsets = 0.2

            # encode feat
            anchor_sort = anchor_decoded[N_start:N_end]
            feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # many [N_num, ?]
            
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)

            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            t_feature_0 = get_time()
            feat_decoded = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = torch.clamp(scale, min=1e-9)

            # use_gated_mlp に応じて分割数とチャネルサイズを変更
            if self.mlp_deform.use_gated:
                num_chunks = 10  # 10分割
                chunk_size = 5   # 各5次元
            else:
                num_chunks = 5   # 5分割
                chunk_size = 10  # 各10次元

            for cc in range(num_chunks):
                # mean_adj, scale_adj, prob_adj, gate = self.get_deform_mlp.forward(feat_decoded, mean_scale, to_dec=cc)
                mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat_decoded, mean_scale, to_dec=cc)
                probs = torch.stack([prob[:, cc*chunk_size:cc*chunk_size+chunk_size], prob_adj], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                Q_feat_tmp = Q_feat[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1)

                feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                    [mean[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                    [scale[:, cc*chunk_size:cc*chunk_size+chunk_size].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_feat_tmp,
                    file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)

                feat_decoded_tmp = feat_decoded_tmp.view(N_num, chunk_size)
                feat_decoded[:, cc*chunk_size:cc*chunk_size+chunk_size] = feat_decoded_tmp
            feat_decoded_list.append(feat_decoded)
            t_feature += get_time() - t_feature_0

            t_scaling_0 = get_time()
            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, chunk_size=10_0000)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)
            t_offset += get_time() - t_offset_0

            xyz_decoded_list.append(anchor_sort)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N, 6], device='cuda')
        _mask = torch.zeros(size=[N, self.n_offsets+1, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N, :10] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        self._anchor_feat = nn.Parameter(_anchor_feat)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        log_info_time = f"\nDecoded time in s: " \
                        f"anchor {round(t_anchor, 4)}, " \
                        f"feat {round(t_feature, 4)}, " \
                        f"scaling {round(t_scaling, 4)}, " \
                        f"offsets {round(t_offset, 4)}, " \
                        f"hash {round(t_hash, 4)}, " \
                        f"masks {round(t_mask, 4)}, " \
                        f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time

        return log_info


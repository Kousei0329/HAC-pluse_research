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

import math
import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2, Entropy_gaussian_mix_prob_3

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt
    
class GEGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        v, g = self.fc(x).chunk(2, dim=-1)
        return F.gelu(v) * g

class GEGLUAct(nn.Module):
    """Drop-in activation: preceding Linear must output 2x width. Returns half width."""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(x) * gate


def _prune_linear_geglu_pair(lin_prev: nn.Linear, lin_next: nn.Linear, ratio: float, min_keep: int = 8):
    """Structurally removes the least-important GEGLU hidden units between two Linear layers.

    lin_prev outputs 2*H (H value channels followed by H gate channels, per GEGLUAct's chunk
    convention); lin_next consumes the H-wide GEGLUAct output. Importance of unit i is the L1
    norm of lin_next's incoming column i -- a unit whose output barely reaches anything
    downstream is safe to drop regardless of its own incoming weights. Returns freshly-sized
    Linear modules with copied weights (unchanged originals if ratio rounds to zero prunable
    units).
    """
    H = lin_prev.out_features // 2
    assert lin_next.in_features == H, f'expected GEGLU pairing, got {lin_prev.out_features} -> {lin_next.in_features}'
    with torch.no_grad():
        importance = lin_next.weight.abs().sum(dim=0)  # [H]
    keep = max(min_keep, H - int(round(H * ratio)))
    keep = min(keep, H)
    if keep >= H:
        return lin_prev, lin_next, H, H
    keep_idx = torch.argsort(importance, descending=True)[:keep]
    keep_idx = torch.sort(keep_idx).values

    device = lin_prev.weight.device
    new_prev = nn.Linear(lin_prev.in_features, keep * 2).to(device)
    new_next = nn.Linear(keep, lin_next.out_features).to(device)
    with torch.no_grad():
        new_prev.weight.copy_(torch.cat([lin_prev.weight[keep_idx], lin_prev.weight[H:][keep_idx]], dim=0))
        new_prev.bias.copy_(torch.cat([lin_prev.bias[keep_idx], lin_prev.bias[H:][keep_idx]], dim=0))
        new_next.weight.copy_(lin_next.weight[:, keep_idx])
        new_next.bias.copy_(lin_next.bias)
    return new_prev, new_next, H, keep


def _resize_module_to_state_dict_(module: nn.Module, state_dict: dict):
    """Rebuilds any nn.Linear inside `module` whose shape doesn't match the corresponding
    '<dotted.path>.weight' tensor in `state_dict`, so a subsequent load_state_dict succeeds even
    when the checkpoint was saved after structured pruning shrank some Linear layers. No-op for
    any Linear whose shape already matches (e.g. no pruning was applied)."""
    device = next(module.parameters()).device

    def _navigate(root, parts):
        obj = root
        for p in parts:
            obj = obj[int(p)] if isinstance(obj, nn.Sequential) and p.isdigit() else getattr(obj, p)
        return obj

    for key, tensor in state_dict.items():
        if not key.endswith('.weight') or tensor.dim() != 2:
            continue
        parts = key[:-len('.weight')].split('.')
        target = _navigate(module, parts)
        if isinstance(target, nn.Linear) and target.weight.shape != tensor.shape:
            out_f, in_f = tensor.shape
            new_lin = nn.Linear(in_f, out_f).to(device)
            parent = _navigate(module, parts[:-1]) if len(parts) > 1 else module
            last = parts[-1]
            if isinstance(parent, nn.Sequential) and last.isdigit():
                parent[int(last)] = new_lin
            else:
                setattr(parent, last, new_lin)


def _prune_sequential_geglu_(seq: nn.Sequential, ratio: float):
    """Prunes every Linear-GEGLUAct-Linear stage inside an nn.Sequential in place, stepping by 2
    so each interior Linear is treated as `prev` once and `next` once (its in/out width can
    shrink independently on each side). Returns (params_before, params_after)."""
    total_before = sum(p.numel() for p in seq.parameters())
    n = len(seq)
    i = 0
    while i + 2 < n:
        if isinstance(seq[i], nn.Linear) and isinstance(seq[i + 1], GEGLUAct) and isinstance(seq[i + 2], nn.Linear):
            new_prev, new_next, _, _ = _prune_linear_geglu_pair(seq[i], seq[i + 2], ratio)
            seq[i] = new_prev
            seq[i + 2] = new_next
        i += 2
    total_after = sum(p.numel() for p in seq.parameters())
    return total_before, total_after
        
class LevelSEGate(nn.Module):
    """SE-Net style content-based gate over hash-grid resolution levels.

    Naive concat lets the downstream MLP know "which position = which level"
    only implicitly (via fixed weight columns), and gives it no signal about
    how the levels relate to one another in scale. This module explicitly
    summarizes each level's own activation, mixes that summary across levels
    (so e.g. an anomalously active fine level can be suppressed relative to
    its coarser neighbors) together with each level's known resolution, and
    rescales each level before concatenation. Output dim is unchanged.
    """
    def __init__(self, n_levels, n_features, resolutions_list, reduction=2):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        log_res = torch.log(torch.tensor(resolutions_list, dtype=torch.float32))
        log_res = (log_res - log_res.mean()) / (log_res.std() + 1e-6)
        self.register_buffer('log_res', log_res)  # [n_levels], fixed, not learned

        hidden = max(n_levels // reduction, 4)
        self.excite = nn.Sequential(
            nn.Linear(n_levels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_levels),
        )
        # start close to identity (gate ~= sigmoid(2) ~= 0.88 for every level, input-independent)
        nn.init.zeros_(self.excite[-1].weight)
        nn.init.constant_(self.excite[-1].bias, 2.0)

    def forward(self, feat):
        # feat: [N, n_levels * n_features]
        N = feat.shape[0]
        feat_l = feat.view(N, self.n_levels, self.n_features)
        squeeze = feat_l.mean(dim=-1)  # [N, n_levels], content-dependent
        res = self.log_res.unsqueeze(0).expand(N, -1)  # [N, n_levels], fixed
        gate = torch.sigmoid(self.excite(torch.cat([squeeze, res], dim=-1)))  # [N, n_levels]
        # Cached (no grad, cheap) so callers can check whether this has learned anything beyond
        # its near-identity init (gate ~= 0.881 for every level, input-independent) -- see
        # gaussian_renderer/__init__.py's periodic debug print.
        self._last_gate_mean = gate.mean(dim=0).detach()
        self._last_gate_std = gate.std(dim=0).detach()
        feat_l = feat_l * gate.unsqueeze(-1)
        return feat_l.reshape(N, self.n_levels * self.n_features)


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
            plane_fusion='concat',
            use_level_gate=False,
    ):
        super().__init__()
        self.use_level_gate = use_level_gate
        assert plane_fusion in ('concat', 'hadamard', 'sum'), \
            f'unknown plane_fusion: {plane_fusion}'
        self.plane_fusion = plane_fusion
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
        assert self.encoding_xy.output_dim == self.encoding_xz.output_dim == self.encoding_yz.output_dim, \
            'xy/xz/yz planes must share the same output_dim to be fused via hadamard/sum'
        if plane_fusion == 'concat':
            planes_dim = self.encoding_xy.output_dim + self.encoding_xz.output_dim + self.encoding_yz.output_dim
        else:
            # hadamard / sum fuse the 3 orthogonal planes into a single feature of one plane's width,
            # following the tri-plane fusion used by K-Planes (hadamard) / EG3D (sum) instead of concat.
            planes_dim = self.encoding_xy.output_dim
        self.output_dim = self.encoding_xyz.output_dim + planes_dim

        if use_level_gate:
            self.gate_xyz = LevelSEGate(self.encoding_xyz.n_levels, n_features, resolutions_list)
            self.gate_xy = LevelSEGate(self.encoding_xy.n_levels, n_features, resolutions_list_2D)
            self.gate_xz = LevelSEGate(self.encoding_xz.n_levels, n_features, resolutions_list_2D)
            self.gate_yz = LevelSEGate(self.encoding_yz.n_levels, n_features, resolutions_list_2D)

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        if self.use_level_gate:
            out_xyz = self.gate_xyz(out_xyz)
            out_xy = self.gate_xy(out_xy)
            out_xz = self.gate_xz(out_xz)
            out_yz = self.gate_yz(out_yz)
        if self.plane_fusion == 'hadamard':
            out_planes = out_xy * out_xz * out_yz
        elif self.plane_fusion == 'sum':
            out_planes = out_xy + out_xz + out_yz
        else:
            out_planes = torch.cat([out_xy, out_xz, out_yz], dim=-1)
        out_i = torch.cat([out_xyz, out_planes], dim=-1)  # [..., 56] when concat, smaller otherwise
        return out_i

class Channel_CTX_fea(nn.Module):
    """Autoregressive per-channel-group context for feat's entropy model.

    n_mix=2 (default, original behavior): outputs one adjustment set (mean_adj, scale_adj,
    prob_adj), combined with mlp_grid's own (mean, scale, prob) into a 2-component mixture.
    n_mix=3: outputs two adjustment sets, for a 3-component mixture (Entropy_gaussian_mix_prob_3),
    using the same autoregressive context (Entropy_gaussian_mix_prob_3 is already implemented in
    utils/entropy_models.py but was never wired up to a source of a 3rd component before).
    """
    def __init__(self, n_mix: int = 2):
        super().__init__()
        assert n_mix in (2, 3), f'Channel_CTX_fea only supports n_mix in (2, 3), got {n_mix}'
        self.n_mix = n_mix
        n_adj = n_mix - 1  # number of (mean,scale,prob) adjustment sets this module produces
        out_per_group = 10 * 3 * n_adj
        # Widened (2x hidden) and deepened (+1 hidden layer), matching mlp_grid's capacity bump.
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50*3+10*0, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, out_per_group),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50*3+10*1, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, out_per_group),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50*3+10*2, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, out_per_group),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50*3+10*3, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, out_per_group),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50*3+10*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, 20*8),
            GEGLUAct(),
            nn.Linear(20*4, out_per_group),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        n_adj = self.n_mix - 1
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        o0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3*n_adj, dim=-1)
        o1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3*n_adj, dim=-1)
        o2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3*n_adj, dim=-1)
        o3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3*n_adj, dim=-1)
        o4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3*n_adj, dim=-1)
        outs = [o0, o1, o2, o3, o4]

        if to_dec in (0, 1, 2, 3, 4):
            return outs[to_dec]  # (mean_adj[, mean_adj2], scale_adj[, scale_adj2], prob_adj[, prob_adj2])

        # Concatenate each of the 3*n_adj output slots across all 5 groups.
        return tuple(torch.cat([o[i] for o in outs], dim=-1) for i in range(3*n_adj))

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*6),
            GEGLUAct(),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*6),
            GEGLUAct(),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*6),
            GEGLUAct(),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*6),
            GEGLUAct(),
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

class CausalKNNContext(nn.Module):
    """Morton順ソート済みアンカー列に対して因果的K近傍集約を行う。

    改善点:
    - クロスチャンク lookback: 前チャンクの直近 max_lookback 件も参照
    - 学習可能温度: log_temperature パラメータで距離重みを適応的に調整
    """
    def __init__(self, hash_dim: int, K: int = 16, temperature: float = 1.0,
                 max_lookback: int = 3000, hidden_mult: int = 8):
        super().__init__()
        self.K = K
        self.max_lookback = max_lookback
        # exp(log_temperature) で常に正、初期値 = temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32))
        # hash_feats を正規化してから correction を計算（学習安定化）
        self.norm = nn.LayerNorm(hash_dim)
        # 残差補正: 3層 + LayerNorm + GELU で高い表現力を確保。hidden_mult(既定8, 2x幅+1層相当)は
        # このネットワーク自体のパラメータ数(=送信が必要なサイズ)を直接支配するので、
        # get_mlp_size()がcausal_knnを正しく計上するようになった後は、baseline相当のサイズに
        # 抑えたい場合はここを2〜4に下げるのが効果的(モデル容量とサイズのトレードオフ)。
        assert hidden_mult % 2 == 0, "hidden_mult must be even (GEGLUAct halves it between layers)"
        h = hash_dim * hidden_mult
        h_half = hash_dim * (hidden_mult // 2)
        self.correction = nn.Sequential(
            nn.Linear(hash_dim * 2, h),
            nn.LayerNorm(h),
            GEGLUAct(),
            nn.Linear(h_half, h),
            nn.LayerNorm(h),
            GEGLUAct(),
            nn.Linear(h_half, hash_dim),
        )

    def _get_temp(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-2, max=10.0)

    def _aggregate(self, query_pos: torch.Tensor, key_pos: torch.Tensor,
                   key_feats: torch.Tensor) -> torch.Tensor:
        """query_pos の各点に対して key_pos の中から K近傍を温度付き softmax で集約。"""
        M = key_pos.shape[0]
        if M == 0:
            return torch.zeros(query_pos.shape[0], key_feats.shape[-1],
                               device=query_pos.device)
        K_eff = min(self.K, M)
        temp = self._get_temp()
        dists = torch.cdist(query_pos, key_pos)                          # [C, M]
        topk_dists, topk_idx = dists.topk(K_eff, largest=False, dim=-1) # [C, K]
        w = torch.softmax(-topk_dists / temp, dim=-1)                   # [C, K]
        return (key_feats[topk_idx] * w.unsqueeze(-1)).sum(1)           # [C, D]

    def _aggregate_causal(self, hash_feats: torch.Tensor, anchor_pos: torch.Tensor,
                          chunk_size: int) -> torch.Tensor:
        """クロスチャンク lookback + チャンク内因果集約。fusion は適用しない。"""
        N, D = hash_feats.shape
        causal_ctx = torch.zeros_like(hash_feats)
        temp = self._get_temp()

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            C = e - s
            chunk_pos   = anchor_pos[s:e]
            chunk_feats = hash_feats[s:e]

            # ── クロスチャンク: 直近 max_lookback 件から集約 ──
            if s > 0:
                prev_start = max(0, s - self.max_lookback)
                ctx_prev = self._aggregate(
                    chunk_pos,
                    anchor_pos[prev_start:s],
                    hash_feats[prev_start:s],
                )  # [C, D]
            else:
                ctx_prev = torch.zeros(C, D, device=hash_feats.device)

            # ── チャンク内因果集約 (上三角マスクで自分より後ろを除外) ──
            if C > 1:
                dists = torch.cdist(chunk_pos, chunk_pos)  # [C, C]
                causal_mask = torch.triu(
                    torch.ones(C, C, dtype=torch.bool, device=hash_feats.device),
                    diagonal=0)
                dists = dists.masked_fill(causal_mask, float('inf'))
                K_eff = min(self.K, C - 1)
                topk_dists, topk_idx = dists.topk(K_eff, largest=False, dim=-1)
                valid = topk_dists < 1e9
                w = torch.where(
                    valid,
                    torch.softmax(
                        torch.where(valid, -topk_dists / temp,
                                    torch.full_like(topk_dists, -1e9)), dim=-1),
                    torch.zeros_like(topk_dists))
                ctx_intra = (chunk_feats[topk_idx] * w.unsqueeze(-1)).sum(1)
            else:
                ctx_intra = torch.zeros(C, D, device=hash_feats.device)

            # クロスチャンクとチャンク内を平均
            has_prev  = s > 0
            has_intra = C > 1
            if has_prev and has_intra:
                causal_ctx[s:e] = (ctx_prev + ctx_intra) / 2
            elif has_prev:
                causal_ctx[s:e] = ctx_prev
            else:
                causal_ctx[s:e] = ctx_intra

        return causal_ctx

    def apply_fusion(self, hash_feats: torch.Tensor, causal_ctx: torch.Tensor) -> torch.Tensor:
        """残差 + correction で hash_feats を精製する。renderer から直接呼ばれる。
        output = hash_feats + correction(cat([norm(hash_feats), causal_ctx]))
        """
        return hash_feats + self.correction(
            torch.cat([self.norm(hash_feats), causal_ctx], dim=-1))

    def aggregate_only(self, hash_feats: torch.Tensor, anchor_pos: torch.Tensor,
                       chunk_size: int = 3000) -> torch.Tensor:
        """Fusion を適用せず causal aggregation のみ行う（renderer から呼ばれる）。"""
        return self._aggregate_causal(hash_feats, anchor_pos, chunk_size)

    def forward(self, hash_feats: torch.Tensor, anchor_pos: torch.Tensor,
                chunk_size: int = 3000) -> torch.Tensor:
        """hash_feats / anchor_pos は Morton 順ソート済みであること。"""
        causal_ctx = self._aggregate_causal(hash_feats, anchor_pos, chunk_size)
        # apply_fusion をチャンク処理してOOMを防ぐ
        N = hash_feats.shape[0]
        out = torch.empty_like(hash_feats)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            out[s:e] = self.apply_fusion(hash_feats[s:e], causal_ctx[s:e])
        return out


class AnchorCondNorm(nn.Module):
    """Three-stage FiLM normalization conditioned on anchor scale, offset, and feature.
    各ステージは対応する入力が None のときスキップされる。

    Stage 1 (scale):  h1  = (1 + γ_s) * LayerNorm(x)  + β_s   (anchor_scale が None なら恒等)
    Stage 2 (offset): h2  = (1 + γ_o) * LayerNorm(h1) + β_o   (anchor_offset が None なら恒等)
    Stage 3 (feat):   out = (1 + γ_f) * LayerNorm(h2) + β_f   (anchor_feat が None なら恒等)
    """
    def __init__(self, feat_dim: int, scale_cond_dim: int, offset_cond_dim: int, anchor_feat_dim: int):
        super().__init__()
        self.norm_scale = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.norm_offset = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.norm_feat = nn.LayerNorm(feat_dim, elementwise_affine=False)
        # Widened (2x hidden) and deepened (+1 hidden layer), matching mlp_grid's capacity bump.
        self.scale_proj = nn.Sequential(
            nn.Linear(scale_cond_dim, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        self.offset_proj = nn.Sequential(
            nn.Linear(offset_cond_dim, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(anchor_feat_dim, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 4),
            GEGLUAct(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor,
                anchor_scale: torch.Tensor = None,
                anchor_offset_flat: torch.Tensor = None,
                anchor_feat: torch.Tensor = None) -> torch.Tensor:
        h = x
        if anchor_scale is not None:
            gamma_s, beta_s = self.scale_proj(anchor_scale).chunk(2, dim=-1)
            h = (1 + gamma_s) * self.norm_scale(h) + beta_s
        if anchor_offset_flat is not None:
            gamma_o, beta_o = self.offset_proj(anchor_offset_flat).chunk(2, dim=-1)
            h = (1 + gamma_o) * self.norm_offset(h) + beta_o
        if anchor_feat is not None:
            gamma_f, beta_f = self.feat_proj(anchor_feat).chunk(2, dim=-1)
            h = (1 + gamma_f) * self.norm_feat(h) + beta_f
        return h


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
                 plane_fusion: str='concat',
                 use_level_gate: bool=False,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 use_gated_mlp: bool=False,
                 use_spatial_context: bool=False,
                 use_joint_context: bool=False,
                 use_anchor_cond_norm: bool=False,
                 use_causal_knn: bool=False,
                 causal_knn_K: int=16,
                 causal_knn_hidden_mult: int=8,
                 mlp_grid_hidden_mult: int=8,
                 use_3gmm: bool=False,
                 use_reno: bool=False,
                 reno_ckpt_path: str=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   'submodules', 'reno', 'model', 'Ford', 'ckpt.pt'),
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise, 'plane_fusion=', plane_fusion,
              'use_level_gate=', use_level_gate)

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
        self.plane_fusion = plane_fusion
        self.use_level_gate = use_level_gate
        self.decoded_version = decoded_version
        self.use_gated_mlp = use_gated_mlp
        self.use_spatial_context = use_spatial_context
        self.use_joint_context = use_joint_context
        self.use_anchor_cond_norm = use_anchor_cond_norm
        self.use_causal_knn = use_causal_knn
        self.use_3gmm = use_3gmm
        self.use_reno = use_reno
        self.reno_ckpt_path = reno_ckpt_path

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
                plane_fusion=plane_fusion,
                use_level_gate=use_level_gate,
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
                nn.Linear(3+1, feat_dim*2),
                GEGLUAct(),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim*2),
            GEGLUAct(),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim*2),
            GEGLUAct(),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim*2),
            GEGLUAct(),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        # Entropy-prediction MLP: widened (2x hidden width) and deepened (+1 hidden layer)
        # relative to the original 2-hidden-layer design, so it has more capacity to exploit
        # the causal_knn / anchor_cond_norm context that feeds it. mlp_grid_hidden_mult
        # (default 8 = the widened design) directly controls this network's own parameter
        # count/transmitted size -- lower it (e.g. 4) to trade capacity for a smaller MLPs size.
        assert mlp_grid_hidden_mult % 2 == 0, "mlp_grid_hidden_mult must be even (GEGLUAct halves it between layers)"
        g_h = feat_dim * mlp_grid_hidden_mult
        g_h_half = feat_dim * (mlp_grid_hidden_mult // 2)
        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, g_h),
            GEGLUAct(),
            nn.Linear(g_h_half, g_h),
            GEGLUAct(),
            nn.Linear(g_h_half, g_h),
            GEGLUAct(),
            nn.Linear(g_h_half, (feat_dim+6+3*self.n_offsets)*2+feat_dim+1+1+1),
        ).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea(n_mix=3 if use_3gmm else 2).cuda()
        else:
            assert not use_3gmm, 'use_3gmm is not implemented for Channel_CTX_fea_tiny (synthetic nerf scenes)'
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        if use_anchor_cond_norm:
            self.anchor_cond_norm = AnchorCondNorm(
                feat_dim=self.encoding_xyz.output_dim,
                scale_cond_dim=6,
                offset_cond_dim=self.n_offsets * 3,
                anchor_feat_dim=feat_dim,
            ).cuda()

        if use_causal_knn:
            self.causal_knn = CausalKNNContext(
                hash_dim=self.encoding_xyz.output_dim,
                K=causal_knn_K,
                temperature=1.0,
                hidden_mult=causal_knn_hidden_mult,
                max_lookback=3000,
            ).cuda()

        import os as _os
        if _os.environ.get('HAC_DEBUG_PARAM_COUNTS'):
            print(f"[DEBUG_PARAM_COUNTS] encoding_xyz.output_dim={self.encoding_xyz.output_dim}")
            print(f"[DEBUG_PARAM_COUNTS] mlp_grid params={sum(p.numel() for p in self.mlp_grid.parameters())}")
            print(f"[DEBUG_PARAM_COUNTS] mlp_deform params={sum(p.numel() for p in self.mlp_deform.parameters())}")
            if use_causal_knn:
                print(f"[DEBUG_PARAM_COUNTS] causal_knn params={sum(p.numel() for p in self.causal_knn.parameters())}")

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()
        self.EG_mix_prob_3 = Entropy_gaussian_mix_prob_3(Q=1).cuda()

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
        quant_bits = getattr(self, '_mlp_quant_bits', None)
        is_fp16 = getattr(self, '_mlp_fp16', False)
        is_fp8 = getattr(self, '_mlp_fp8', False)
        is_fp4 = getattr(self, '_mlp_fp4', False)
        for n, p in self.named_parameters():
            # causal_knn's weights are side info a real decoder needs too (it's not named with
            # 'mlp' so it was previously skipped by this loop entirely -- silently free in every
            # size report). Count it alongside mlp_grid/mlp_deform as an entropy-side-info module.
            is_causal_knn = n.startswith('causal_knn.')
            if not ('mlp' in n or is_causal_knn):
                continue
            # Only mlp_grid/mlp_deform/causal_knn (the entropy-prediction side info) are ever
            # fake-quantized by quantize_mlps_()/quantize_mlps_fp16_()/quantize_mlps_fp8_()/
            # quantize_mlps_fp4_(); everything else (mlp_opacity/cov/color/featurebank/hyp) stays
            # at the full `digit` (fp32) width. All four quantization paths now share the same
            # target set (_entropy_side_info_modules) so int vs float comparisons are apples to
            # apples (same modules quantized either way).
            is_quant_target = is_causal_knn or 'mlp_grid' in n or 'mlp_deform' in n
            if is_fp8 and is_quant_target:
                # True 8-bit float (e4m3/e5m2): no extra scale metadata needed, flat 8 bits/param.
                mlp_size += p.numel()*8
            elif is_fp4 and is_quant_target:
                # 4-bit float (e2m1): needs the same per-output-channel scale as the int path
                # (see quantize_mlps_fp4_ -- e2m1's raw range is too narrow for real weights
                # without one), so charge that scale overhead too, not just the flat 4 bits/param.
                mlp_size += p.numel()*4
                if p.dim() >= 2:
                    mlp_size += p.shape[0]*16
            elif is_fp16 and is_quant_target:
                # True IEEE half precision: no extra scale metadata needed, unlike the linear
                # int quantization below, so it's a flat 16 bits/param (weights and biases).
                mlp_size += p.numel()*16
            elif quant_bits is not None and p.dim() >= 2 and is_quant_target:
                mlp_size += p.numel()*quant_bits
                # Per-output-channel scale overhead (fp16 per row) -- quantize_mlps_ uses one
                # scale per out_feature row, not one per tensor, so this has to be counted too.
                mlp_size += p.shape[0]*16
            else:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def structured_prune_mlps_(self, ratio=0.3):
        """Structurally prunes the least-important GEGLU hidden units out of mlp_grid and every
        Channel_CTX_fea sub-MLP (mlp_deform's MLP_d0..d4), shrinking the actual weight matrices
        (not just zeroing them) so get_mlp_size() reflects fewer parameters directly. Complements
        quantize_mlps_ (fewer params x fewer bits each) -- call this first, then quantize the
        now-smaller network. No fine-tuning after pruning: this is a one-shot post-training cut,
        so expect some quality loss at high ratios.
        """
        total_before = 0
        total_after = 0
        b, a = _prune_sequential_geglu_(self.mlp_grid, ratio)
        total_before += b; total_after += a
        for name in ['MLP_d0', 'MLP_d1', 'MLP_d2', 'MLP_d3', 'MLP_d4']:
            seq = getattr(self.mlp_deform, name, None)
            if seq is not None:
                b, a = _prune_sequential_geglu_(seq, ratio)
                total_before += b; total_after += a
        pct = 100 * (1 - total_after / total_before) if total_before > 0 else 0.0
        print(f"[structured_prune_mlps_] ratio={ratio}: {total_before} -> {total_after} params ({pct:.1f}% reduction)")

    def diagnose_feat_calibration(self):
        """One-shot diagnostic: how well does feat's predicted mixture distribution match the
        actual (already-quantized) feat values it has to encode? Mirrors estimate_final_bits()'s
        feat pipeline exactly, then reports the calibration gap: actual residual spread vs the
        model's own predicted spread, and the fraction of |z|>3 outliers against the dominant
        mixture component vs the ~0.27% a well-calibrated Gaussian would produce. actual >>
        predicted means the predicted scale is too narrow (arithmetic coding pays a
        quadratic-in-residual penalty for this -- the likely cause of a post-pruning bit-cost
        blowup); actual << predicted means it's too wide (bits being left on the table).
        """
        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]
        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]

        if self.use_causal_knn:
            _anchor_int = torch.round(_anchor / self.voxel_size).long()
            sorted_indices = calculate_morton_order(_anchor_int)
            unsort_indices = torch.argsort(sorted_indices)
            anchor_sorted = _anchor[sorted_indices]
            hash_feats = self.calc_interp_feat(anchor_sorted)
            causal_ctx = self.causal_knn.aggregate_only(hash_feats, anchor_sorted, chunk_size=MAX_batch_size)
            feat_context_no_cond = self.causal_knn.apply_fusion(hash_feats, causal_ctx)[unsort_indices]
        else:
            feat_context_no_cond = self.calc_interp_feat(_anchor)

        _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj_no_cond, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context_no_cond), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
        Q_scaling = torch.clamp(0.001 * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
        Q_offsets = torch.clamp(0.2 * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling, self.get_scaling.mean())).detach()
        offsets_full = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1), self._offset.mean())).detach()
        offsets_full = offsets_full * _mask.repeat(1, 1, 3).to(offsets_full.dtype)

        if self.use_causal_knn:
            hash_feats_cond = self.calc_interp_feat(anchor_sorted, grid_scaling[sorted_indices], offsets_full[sorted_indices])
            feat_context_with_scale_offset = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx)[unsort_indices]
        else:
            feat_context_with_scale_offset = self.calc_interp_feat(_anchor, grid_scaling, offsets_full)

        mean, scale, prob, _, _, _, _, _, _, _ = \
            torch.split(self.get_grid_mlp(feat_context_with_scale_offset), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
        Q_feat_adj = Q_feat_adj_no_cond
        Q_feat = torch.clamp(1 * (1 + torch.tanh(Q_feat_adj)), min=0.0001)
        _feat = (STE_multistep.apply(_feat, Q_feat, self._anchor_feat.mean())).detach()

        mean_list, scale_list, probs_list = self.get_feat_mixture(_feat, mean, scale, prob)

        means = torch.stack(mean_list, dim=0)   # [K, N, D]
        scales = torch.stack(scale_list, dim=0).clamp(min=1e-6)
        probs = torch.stack(probs_list, dim=0)
        probs = probs / probs.sum(dim=0, keepdim=True).clamp(min=1e-8)

        weighted_mean = (probs * means).sum(dim=0)
        weighted_scale = (probs * scales).sum(dim=0)
        residual = _feat - weighted_mean
        actual_std = residual.std().item()
        pred_scale_mean = weighted_scale.mean().item()

        dominant = torch.argmax(probs, dim=0)
        dom_mean = torch.gather(means, 0, dominant.unsqueeze(0)).squeeze(0)
        dom_scale = torch.gather(scales, 0, dominant.unsqueeze(0)).squeeze(0).clamp(min=1e-6)
        z = (_feat - dom_mean) / dom_scale
        outlier_frac = (z.abs() > 3).float().mean().item()
        expected_outlier_frac = 2 * (1 - 0.5 * (1 + math.erf(3 / math.sqrt(2))))

        print(f"[diagnose_feat_calibration] N={_feat.shape[0]} D={_feat.shape[1]} n_mix={probs.shape[0]}")
        print(f"[diagnose_feat_calibration] actual residual std={actual_std:.4f} vs predicted avg scale={pred_scale_mean:.4f} "
              f"(ratio={actual_std/max(pred_scale_mean,1e-8):.3f}; >1 => predicted scale too narrow, <1 => too wide)")
        print(f"[diagnose_feat_calibration] dominant-component z: mean={z.mean().item():.4f} std={z.std().item():.4f} "
              f"|z|>3 frac: actual={outlier_frac*100:.3f}% vs well-calibrated~{expected_outlier_frac*100:.3f}%")

    def finetune_pruned_mlps_(self, iters=300, lr=1e-4, chunk_size=50_000, lambda_q_reg=1000.0):
        """Brief post-prune recovery: re-fits mlp_grid/mlp_deform's entropy predictions (mean,
        scale, mixture prob, Q) to the already-fixed anchor/feat/scaling/offset values by
        directly minimizing the same rate loss estimate_final_bits() reports. Only mlp_grid and
        mlp_deform receive gradients -- geometry/appearance and every other network stay frozen.
        structured_prune_mlps_'s magnitude heuristic has no gradient signal, so without this a
        pruned entropy head can leave the real arithmetic encoder/decoder round-trip badly
        miscalibrated for outlier anchors (see diagnose_feat_calibration).

        Processes anchors in `chunk_size` pieces with gradient accumulation (chunk_loss.backward()
        called per chunk, one optimizer.step() per full pass) -- at full scene scale (300k+
        anchors) holding the whole set's activations for backward at once OOMs even though the
        parameter count being trained is tiny; chunking bounds peak memory to one chunk while
        still computing the exact full-batch gradient (sum of per-chunk gradients). calc_interp_feat
        and causal_knn.apply_fusion are pure per-anchor operations (no cross-anchor mixing), so
        chunking in original anchor order is exact -- only the causal aggregation into causal_ctx
        needs the full sorted set, and that's precomputed once, unsorted back to original order,
        before chunking begins.

        Q_scaling/Q_offsets/Q_feat (the quantization step sizes) come out of the SAME mlp_grid
        heads as mean/scale/prob, but unlike those, they directly set how coarsely _scaling/
        _offset/_feat get quantized via STE -- i.e. they control reconstruction fidelity, not just
        bit cost. A rate-only loss has zero incentive to keep them where they were and every
        incentive to push them larger (coarser = fewer bits), which taxes PSNR for a rate-side
        fix. lambda_q_reg penalizes squared relative drift of each Q away from its immediately
        post-prune (pre-finetune) value, letting mean/scale/prob (pure entropy-modeling, no
        reconstruction impact) keep improving while holding quantization granularity roughly
        fixed.
        """
        Q_feat_init, Q_scaling_init, Q_offsets_init = 1, 0.001, 0.2

        params = list(self.mlp_grid.parameters()) + list(self.mlp_deform.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        EG = self.EG_mix_prob_3 if self.use_3gmm else self.EG_mix_prob_2

        with torch.no_grad():
            mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]
            _anchor = self.get_anchor[mask_anchor].detach()
            _feat_raw = self._anchor_feat[mask_anchor].detach()
            _grid_offsets_raw = self._offset[mask_anchor].detach()
            _scaling_raw = self.get_scaling[mask_anchor].detach()
            _mask = self.get_mask[mask_anchor].detach()
            feat_mean_ref = self._anchor_feat.mean().detach()
            scaling_mean_ref = self.get_scaling.mean().detach()
            offset_mean_ref = self._offset.mean().detach()

            if self.use_causal_knn:
                _anchor_int = torch.round(_anchor / self.voxel_size).long()
                sorted_indices = calculate_morton_order(_anchor_int)
                unsort_indices = torch.argsort(sorted_indices)
                anchor_sorted = _anchor[sorted_indices]
                hash_feats = self.calc_interp_feat(anchor_sorted).detach()
                causal_ctx_sorted = self.causal_knn.aggregate_only(hash_feats, anchor_sorted, chunk_size=MAX_batch_size).detach()
                # Constant across iterations: hash_feats/causal_ctx/causal_knn.correction are
                # all fixed (only mlp_grid/mlp_deform are being trained here). Unsorted back to
                # original anchor order once here so the per-iteration loop can chunk in that
                # order directly (apply_fusion/calc_interp_feat are pure per-anchor ops, so this
                # is exact -- no re-sorting needed per chunk).
                feat_context_no_cond = self.causal_knn.apply_fusion(hash_feats, causal_ctx_sorted)[unsort_indices].detach()
                causal_ctx = causal_ctx_sorted[unsort_indices].detach()
            else:
                feat_context_no_cond = self.calc_interp_feat(_anchor).detach()

        N = _anchor.shape[0]
        chunk_size = min(chunk_size, N)
        n_chunks = (N + chunk_size - 1) // chunk_size

        with torch.no_grad():
            # Reference Q's, frozen at their just-pruned (pre-finetune) values. All three come
            # out of the same "no_cond" pass (Q_feat_adj is deliberately sourced from this stage
            # everywhere in the codebase, never the FiLM-conditioned one), so one chunked forward
            # pass here is enough for all of them.
            Q_scaling_ref_list, Q_offsets_ref_list, Q_feat_ref_list = [], [], []
            for c in range(n_chunks):
                s, e = c * chunk_size, min((c + 1) * chunk_size, N)
                _, _, _, _, _, _, _, Q_feat_adj_ref, Q_scaling_adj_ref, Q_offsets_adj_ref = \
                    torch.split(self.get_grid_mlp(feat_context_no_cond[s:e]), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
                Q_scaling_ref_list.append(torch.clamp(Q_scaling_init * (1 + torch.tanh(Q_scaling_adj_ref)), min=0.00001))
                Q_offsets_ref_list.append(torch.clamp(Q_offsets_init * (1 + torch.tanh(Q_offsets_adj_ref)), min=0.0001))
                Q_feat_ref_list.append(torch.clamp(Q_feat_init * (1 + torch.tanh(Q_feat_adj_ref)), min=0.0001))
            Q_scaling_ref = torch.cat(Q_scaling_ref_list, dim=0).detach()
            Q_offsets_ref = torch.cat(Q_offsets_ref_list, dim=0).detach()
            Q_feat_ref = torch.cat(Q_feat_ref_list, dim=0).detach()

        for it in range(iters):
          total_loss_val = 0.0
          with torch.enable_grad():
            # training_report() (this method's only caller) runs inside an outer
            # torch.no_grad() block, so the forward/backward pass here needs an explicit
            # enable_grad() to get a real graph -- otherwise loss.backward() has nothing to
            # walk ("does not require grad and does not have a grad_fn").
            optimizer.zero_grad()

            for c in range(n_chunks):
                s, e = c * chunk_size, min((c + 1) * chunk_size, N)

                _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj_no_cond, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(self.get_grid_mlp(feat_context_no_cond[s:e]), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
                Q_scaling = torch.clamp(Q_scaling_init * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
                Q_offsets = torch.clamp(Q_offsets_init * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

                grid_scaling = (STE_multistep.apply(_scaling_raw[s:e], Q_scaling, scaling_mean_ref)).detach()
                offsets_full = (STE_multistep.apply(_grid_offsets_raw[s:e], Q_offsets.unsqueeze(1), offset_mean_ref)).detach()
                offsets_full = offsets_full * _mask[s:e].repeat(1, 1, 3).to(offsets_full.dtype)

                if self.use_causal_knn:
                    hash_feats_cond = self.calc_interp_feat(_anchor[s:e], grid_scaling, offsets_full)
                    feat_context_with_scale_offset = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx[s:e])
                else:
                    feat_context_with_scale_offset = self.calc_interp_feat(_anchor[s:e], grid_scaling, offsets_full)

                mean, scale, prob, _, _, _, _, _, _, _ = \
                    torch.split(self.get_grid_mlp(feat_context_with_scale_offset), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
                Q_feat_adj = Q_feat_adj_no_cond
                Q_feat = torch.clamp(Q_feat_init * (1 + torch.tanh(Q_feat_adj)), min=0.0001)
                _feat = (STE_multistep.apply(_feat_raw[s:e], Q_feat, feat_mean_ref)).detach()

                mean_list, scale_list, probs_list = self.get_feat_mixture(_feat, mean, scale, prob)
                offsets = offsets_full.view(-1, 3*self.n_offsets)
                mask_tmp = _mask[s:e].repeat(1, 1, 3).view(-1, 3*self.n_offsets)

                bit_feat = EG.forward(_feat, *mean_list, *scale_list, *probs_list, Q=Q_feat)
                bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
                bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
                bit_offsets = bit_offsets * mask_tmp

                rate_loss = bit_feat.sum() + bit_scaling.sum() + bit_offsets.sum()
                q_reg = (((Q_scaling - Q_scaling_ref[s:e]) / Q_scaling_ref[s:e]) ** 2).sum() \
                      + (((Q_offsets - Q_offsets_ref[s:e]) / Q_offsets_ref[s:e]) ** 2).sum() \
                      + (((Q_feat - Q_feat_ref[s:e]) / Q_feat_ref[s:e]) ** 2).sum()
                chunk_loss = rate_loss + lambda_q_reg * q_reg
                chunk_loss.backward()
                total_loss_val += chunk_loss.item()

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

          if it == 0 or it == iters - 1 or (it + 1) % max(1, iters // 5) == 0:
                print(f"[finetune_pruned_mlps_] iter {it+1}/{iters} rate_loss(bits)={total_loss_val:.1f}")

    def quantize_mlps_(self, bits=8):
        """Post-training fake-quantizes (quantize then dequantize in place) mlp_grid's,
        mlp_deform's, and (if enabled) causal_knn's weight matrices to `bits` via per-output-
        channel (per-row) symmetric linear quantization. Biases/LayerNorms (1-D params) are left
        at full precision (negligible parameter count). Mutates self in place; get_mlp_size()
        reflects the smaller size afterward via self._mlp_quant_bits.

        Per-*row* (not per-tensor) scale matters specifically for mlp_grid's final layer: it
        packs mean/scale/prob (50 rows each) together with mean_scaling/scale_scaling (only 6
        rows each) and offsets/Q-adjustments in the *same* weight matrix. A single global scale
        is set by whichever output channel has the largest weights -- empirically feat's rows do
        -- which then coarsens every other channel's precision far more than its own dynamic
        range warrants (scaling's real size grew ~20% under per-tensor quantization even though
        feat/MLPs improved). Per-row scales give every output channel its own precision budget
        instead.
        """
        self._mlp_quant_bits = bits
        qmax = 2 ** (bits - 1) - 1
        n_tensors = 0
        with torch.no_grad():
            for m in self._entropy_side_info_modules():
                for name, p in m.named_parameters():
                    if p.dim() >= 2:
                        # p: [out_features, in_features] for nn.Linear -- one scale per out row.
                        scale = p.abs().amax(dim=1, keepdim=True) / qmax
                        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                        q = torch.clamp(torch.round(p / safe_scale), -qmax, qmax)
                        p.copy_(torch.where(scale > 0, q * safe_scale, p))
                        n_tensors += 1
        print(f"[quantize_mlps_] fake-quantized {n_tensors} weight matrices in "
              f"{', '.join(type(m).__name__ for m in self._entropy_side_info_modules())} to {bits} bits (per-output-channel scale)")

    def _entropy_side_info_modules(self):
        """mlp_grid/mlp_deform plus causal_knn (when enabled) -- every network whose weights are
        side information the decoder needs and that quantize_mlps_fp16_/quantize_mlps_fp8_ compress.
        causal_knn's own parameters were previously never counted by get_mlp_size() at all (its
        submodule names don't contain 'mlp'), so its weights were silently free in every size
        report; they're included here and in get_mlp_size()'s name matching so the reported Total
        actually reflects what a real decoder would need to receive.
        """
        mods = [self.mlp_grid, self.mlp_deform]
        if getattr(self, 'use_causal_knn', False) and hasattr(self, 'causal_knn'):
            mods.append(self.causal_knn)
        return mods

    def quantize_mlps_fp16_(self):
        """Post-training round-trips mlp_grid's, mlp_deform's, and (if enabled) causal_knn's
        parameters (weights and biases) through true IEEE half precision (torch.float16) in
        place, simulating real fp16 storage. Unlike quantize_mlps_ (linear int quantization
        needing a per-row scale), fp16 needs no extra metadata -- get_mlp_size() charges a flat
        16 bits/param via self._mlp_fp16.
        """
        self._mlp_fp16 = True
        n_tensors = 0
        with torch.no_grad():
            for m in self._entropy_side_info_modules():
                for name, p in m.named_parameters():
                    p.copy_(p.half().float())
                    n_tensors += 1
        print(f"[quantize_mlps_fp16_] round-tripped {n_tensors} tensors through fp16 "
              f"({', '.join(type(m).__name__ for m in self._entropy_side_info_modules())})")

    @staticmethod
    def _round_to_minifloat(x, exp_bits, mant_bits, bias):
        """Rounds x to the nearest value representable by a minifloat format with `exp_bits`
        exponent bits, `mant_bits` mantissa bits, and the given exponent bias (no subnormals/
        Inf/NaN handling -- irrelevant for finite, non-extreme network weights). Implemented with
        plain tensor ops (no torch.float8_e4m3fn/e5m2 dtype) because the project's training env
        (HAC_plux_env, torch 1.12.1) predates PyTorch's native fp8 dtypes entirely; this reproduces
        the same round-to-nearest-representable-value semantics on any torch version.
        """
        sign = torch.sign(x)
        absx_safe = torch.clamp(x.abs(), min=1e-30)
        exponent = torch.floor(torch.log2(absx_safe))
        max_exp = 2 ** exp_bits - 1 - bias
        min_exp = -bias + 1
        exponent = torch.clamp(exponent, min=min_exp, max=max_exp)
        scale = torch.pow(2.0, exponent)
        mantissa = absx_safe / scale  # in [1, 2)
        step = 2.0 ** (-mant_bits)
        q_mantissa = torch.round((mantissa - 1.0) / step) * step + 1.0
        overflow = q_mantissa >= 2.0  # rounding pushed the mantissa up to the next exponent
        q_mantissa = torch.where(overflow, torch.ones_like(q_mantissa), q_mantissa)
        exponent = torch.where(overflow, exponent + 1, exponent)
        result = sign * q_mantissa * torch.pow(2.0, exponent)
        return torch.where(x.abs() == 0, torch.zeros_like(x), result)

    def quantize_mlps_fp8_(self, fp8_variant='e4m3'):
        """Post-training round-trips mlp_grid's, mlp_deform's, and (if enabled) causal_knn's
        parameters through an 8-bit float representation -- e4m3 (1 sign + 4 exponent + 3
        mantissa bits, bias 7) or e5m2 (1 sign + 5 exponent + 2 mantissa bits, bias 15) -- the
        same round-trip-in-place technique as quantize_mlps_fp16_ but at half the bit width.
        get_mlp_size() charges a flat 8 bits/param via self._mlp_fp8.
        """
        exp_bits, mant_bits, bias = (4, 3, 7) if fp8_variant == 'e4m3' else (5, 2, 15)
        self._mlp_fp8 = True
        n_tensors = 0
        with torch.no_grad():
            for m in self._entropy_side_info_modules():
                for name, p in m.named_parameters():
                    p.copy_(self._round_to_minifloat(p, exp_bits, mant_bits, bias))
                    n_tensors += 1
        print(f"[quantize_mlps_fp8_] round-tripped {n_tensors} tensors through {fp8_variant} "
              f"({', '.join(type(m).__name__ for m in self._entropy_side_info_modules())})")

    def quantize_mlps_fp4_(self):
        """Post-training round-trips mlp_grid's, mlp_deform's, and (if enabled) causal_knn's
        weight matrices through a 4-bit float representation (e2m1: 1 sign + 2 exponent + 1
        mantissa bit, bias 1 -- the OCP MX/NVFP4 E2M1 layout). Unlike quantize_mlps_fp8_/_fp16_,
        this needs a per-output-channel (per-row) scale first: e2m1's raw representable range is
        only [1, 6] in magnitude (min_exp = -bias+1 = 0), while real weight matrices are
        typically << 1 in magnitude -- casting them unscaled would flush nearly everything to the
        same smallest representable value. Scaling each row so its largest weight lands at 6.0
        (mirroring quantize_mlps_'s int path, and how real FP4 formats like NVFP4/MXFP4 are
        always used with a block scale) is what makes this format usable at all. 1-D params
        (biases/LayerNorms) skip scaling (negligible parameter count; direct e2m1 cast is fine
        for them since get_mlp_size doesn't need to special-case a 1-row scale there).
        get_mlp_size() charges 4 bits/param plus the same 16-bit-per-row scale overhead as the
        int path. Expect a much larger quality hit than fp8 even so -- 2 mantissa values per
        exponent (i.e. per row: {scale, 1.5xscale} x sign x {2^0..2^2}) is very coarse; this
        exists to map out where the precision/size tradeoff actually breaks.
        """
        exp_bits, mant_bits, bias = 2, 1, 1
        max_representable = 1.5 * (2.0 ** (2 ** exp_bits - 1 - bias))  # e2m1 -> 1.5 * 2^2 = 6.0
        self._mlp_fp4 = True
        n_tensors = 0
        with torch.no_grad():
            for m in self._entropy_side_info_modules():
                for name, p in m.named_parameters():
                    if p.dim() >= 2:
                        scale = p.abs().amax(dim=1, keepdim=True) / max_representable
                        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                        q = self._round_to_minifloat(p / safe_scale, exp_bits, mant_bits, bias)
                        p.copy_(torch.where(scale > 0, q * safe_scale, p))
                    else:
                        p.copy_(self._round_to_minifloat(p, exp_bits, mant_bits, bias))
                    n_tensors += 1
        print(f"[quantize_mlps_fp4_] round-tripped {n_tensors} tensors through e2m1 fp4 "
              f"(per-row scaled) ({', '.join(type(m).__name__ for m in self._entropy_side_info_modules())})")

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()
        if self.use_causal_knn:
            self.causal_knn.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.use_anchor_cond_norm:
            self.anchor_cond_norm.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()
        if self.use_causal_knn:
            self.causal_knn.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.use_anchor_cond_norm:
            self.anchor_cond_norm.train()

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

    def forward_grid(self, feat_context):
        y = self.mlp_grid(feat_context)
        return torch.split(
            y,
            [self.feat_dim, self.feat_dim, self.feat_dim,
             6, 6,
             3 * self.n_offsets, 3 * self.n_offsets,
             1, 1, 1],
            dim=-1,
        )

    def get_feat_mixture(self, feat_for_ar_ctx, mean, scale, prob, mean_scale_ctx=None, to_dec=-1):
        """Builds the per-component (mean, scale, prob) lists for feat's entropy mixture,
        uniformly handling the 2-component (default) and 3-component (use_3gmm) cases.

        feat_for_ar_ctx: the feat values fed as the autoregressive context to mlp_deform
        (quantized ground truth when encoding, the running decoded buffer when decoding).
        mean/scale/prob: mlp_grid's own base-component prediction for the segment of feat
        actually being entropy-coded this call (the full feat_dim width when to_dec=-1, or just
        the 10-dim slice for group `to_dec` in the chunked per-group encode/decode loops).
        mean_scale_ctx: the *full* (unsliced) cat([mean, scale, prob]) conditioning input
        get_deform_mlp expects regardless of to_dec -- pass this explicitly when mean/scale/prob
        above are already sliced to one group; defaults to cat([mean, scale, prob]) when not
        (i.e. when to_dec=-1 and mean/scale/prob are the full-width tensors already).

        Returns (mean_list, scale_list, probs_list) -- pass unpacked (*mean_list, *scale_list,
        *probs_list) into EG_mix_prob_2/3.forward, or as plain lists into
        encoder_gaussian_mixed_chunk / decoder_gaussian_mixed_chunk (already list-generic).
        """
        if mean_scale_ctx is None:
            mean_scale_ctx = torch.cat([mean, scale, prob], dim=-1)
        if self.use_3gmm:
            mean_adj, scale_adj, prob_adj, mean_adj2, scale_adj2, prob_adj2 = \
                self.get_deform_mlp.forward(feat_for_ar_ctx, mean_scale_ctx, to_dec=to_dec)
            probs = torch.softmax(torch.stack([prob, prob_adj, prob_adj2], dim=-1), dim=-1)
            return [mean, mean_adj, mean_adj2], [scale, scale_adj, scale_adj2], \
                   [probs[..., 0], probs[..., 1], probs[..., 2]]
        else:
            mean_adj, scale_adj, prob_adj = \
                self.get_deform_mlp.forward(feat_for_ar_ctx, mean_scale_ctx, to_dec=to_dec)
            probs = torch.softmax(torch.stack([prob, prob_adj], dim=-1), dim=-1)
            return [mean, mean_adj], [scale, scale_adj], [probs[..., 0], probs[..., 1]]

    def calc_interp_feat(self, x, anchor_scale=None, anchor_offset=None, anchor_feat=None):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, D_hash]
        if self.use_anchor_cond_norm and (anchor_scale is not None or anchor_offset is not None or anchor_feat is not None):
            features = self.anchor_cond_norm(
                features,
                anchor_scale,
                anchor_offset.reshape(anchor_offset.shape[0], -1) if anchor_offset is not None else None,
                anchor_feat,
            )
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

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
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
            if self.use_causal_knn:
                l.append({'params': self.causal_knn.parameters(), 'lr': training_args.causal_knn_lr_init, "name": "causal_knn"})
            if self.use_anchor_cond_norm:
                l.append({'params': self.anchor_cond_norm.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "anchor_cond_norm"})
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
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
            if self.use_causal_knn:
                l.append({'params': self.causal_knn.parameters(), 'lr': training_args.causal_knn_lr_init, "name": "causal_knn"})
            if self.use_anchor_cond_norm:
                l.append({'params': self.anchor_cond_norm.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "anchor_cond_norm"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
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
        if self.use_causal_knn:
            self.causal_knn_scheduler_args = get_expon_lr_func(lr_init=training_args.causal_knn_lr_init,
                                                        lr_final=training_args.causal_knn_lr_final,
                                                        lr_delay_mult=training_args.causal_knn_lr_delay_mult,
                                                        max_steps=training_args.causal_knn_lr_max_steps)

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
            if self.use_anchor_cond_norm and param_group["name"] == "anchor_cond_norm":
                # Shares mlp_grid's schedule since it was initialized with mlp_grid_lr_init and
                # never had its own decay branch -- previously harmless because this module was
                # dead code whenever use_causal_knn=True (see calc_interp_feat call sites), but
                # now that it actually participates in the forward pass its LR must anneal too,
                # or it keeps making large updates for the entire run while everything else has
                # converged, which can destabilize the feat entropy model late in training.
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_causal_knn and param_group["name"] == "causal_knn":
                lr = self.causal_knn_scheduler_args(iteration)
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
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or len(group["params"]) != 1:
                continue
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
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or len(group["params"]) != 1:
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

                if remove_duplicates_list:
                    remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
                else:
                    # grid_coords is empty (self.get_anchor has 0 anchors, e.g. every anchor was
                    # pruned away) -- trivially nothing to deduplicate against, so every candidate
                    # is new. reduce() has no identity element for an empty sequence and would
                    # otherwise raise TypeError here.
                    remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
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

                torch.cuda.empty_cache()
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

        ckpt = {
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'encoding_xyz': self.encoding_xyz.state_dict(),
            'grid_mlp': self.mlp_grid.state_dict(),
            'deform_mlp': self.mlp_deform.state_dict(),
        }
        if self.use_causal_knn:
            ckpt['causal_knn'] = self.causal_knn.state_dict()
        if self.use_feat_bank:
            ckpt['mlp_feature_bank'] = self.mlp_feature_bank.state_dict()
        if self.use_anchor_cond_norm:
            ckpt['anchor_cond_norm'] = self.anchor_cond_norm.state_dict()
        torch.save(ckpt, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        # Rebuild any Linear whose shape was shrunk by structured_prune_mlps_() before this
        # checkpoint was saved -- mlp_grid/mlp_deform are otherwise always constructed at their
        # full (unpruned) size in __init__, so load_state_dict would fail on a size mismatch.
        _resize_module_to_state_dict_(self.mlp_grid, checkpoint['grid_mlp'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        _resize_module_to_state_dict_(self.mlp_deform, checkpoint['deform_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])
        if self.use_causal_knn and 'causal_knn' in checkpoint:
            self.causal_knn.load_state_dict(checkpoint['causal_knn'])
        if self.use_anchor_cond_norm and 'anchor_cond_norm' in checkpoint:
            self.anchor_cond_norm.load_state_dict(checkpoint['anchor_cond_norm'])

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

    def compute_total_bits(self):
        """全アンカーを使って feat / scaling / offsets それぞれの総ビット数を返す。"""
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]

        _anchor_int = torch.round(_anchor / self.voxel_size).long()
        sorted_indices = calculate_morton_order(_anchor_int)
        unsort_indices = torch.argsort(sorted_indices)
        anchor_sorted = _anchor[sorted_indices]
        # Pruned (masked-off) offset slots are never trained against any rendering loss, so their
        # raw value is unconstrained garbage; zero them out before using them as FiLM conditioning
        # input, matching what conduct_encoding()/estimate_final_bits() condition on.
        _grid_offsets_masked = _grid_offsets * _mask.repeat(1, 1, 3).to(_grid_offsets.dtype)

        if self.use_causal_knn:
            # Two-stage, matching the non-causal_knn branch: reuse the same (expensive) neighbor
            # aggregation for both stages, only re-running the cheap calc_interp_feat + apply_fusion
            # so stage 2 can condition on scaling/offset via FiLM (AnchorCondNorm).
            hash_feats = self.calc_interp_feat(anchor_sorted)
            causal_ctx = self.causal_knn.aggregate_only(hash_feats, anchor_sorted, chunk_size=MAX_batch_size)
            feat_context_no_cond = self.causal_knn.apply_fusion(hash_feats, causal_ctx)[unsort_indices]
            hash_feats_cond = self.calc_interp_feat(
                anchor_sorted,
                _scaling[sorted_indices],
                _grid_offsets_masked[sorted_indices],
            )
            feat_context_with_scale_offset = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx)[unsort_indices]
        else:
            feat_context_no_cond = self.calc_interp_feat(anchor_sorted)[unsort_indices]
            feat_context_with_scale_offset = self.calc_interp_feat(
                anchor_sorted,
                _scaling[sorted_indices],
                _grid_offsets_masked[sorted_indices],
            )[unsort_indices]

        # MLP + entropy を chunk で処理して OOM を回避
        # feat_context / feat_context_{no_cond,with_scale_offset} は全体を事前計算済み
        # (causal_knn は全アンカー依存のため chunking 不可) → それ以降だけ分割
        _Q_feat_base    = Q_feat      # scalar
        _Q_scaling_base = Q_scaling   # scalar
        _Q_offsets_base = Q_offsets   # scalar

        N = _feat.shape[0]
        CHUNK = MAX_batch_size  # reuse the same batch-size knob
        total_feat_bits = 0.0
        total_scaling_bits = 0.0
        total_offsets_bits = 0.0

        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)

            feat_c     = _feat[s:e]
            scaling_c  = _scaling[s:e]
            offsets_c  = _grid_offsets[s:e]
            mask_c     = _mask[s:e]

            _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj_no_cond, Q_scaling_adj, Q_offsets_adj = \
                self.forward_grid(feat_context_no_cond[s:e])
            mean, scale, prob, _, _, _, _, _, _, _ = \
                self.forward_grid(feat_context_with_scale_offset[s:e])
            # See gaussian_renderer/__init__.py for why: Q_feat_adj always comes from the
            # unconditioned context, never the FiLM-conditioned one (causal_knn and non-causal_knn
            # both shown to destabilize when it's sourced from the conditioned pass).
            Q_feat_adj = Q_feat_adj_no_cond

            Q_feat_adj    = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            Qf = torch.clamp(_Q_feat_base    * (1 + torch.tanh(Q_feat_adj)),    min=1e-4)
            Qs = torch.clamp(_Q_scaling_base * (1 + torch.tanh(Q_scaling_adj)), min=1e-5)
            Qo = torch.clamp(_Q_offsets_base * (1 + torch.tanh(Q_offsets_adj)), min=1e-4)

            feat_q = (STE_multistep.apply(feat_c, Qf)).detach()
            mean_list, scale_list, probs_list = self.get_feat_mixture(feat_q, mean, scale, prob)

            mean_s  = mean_scaling.contiguous().view(-1)
            scale_s = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-3)
            mean_o  = mean_offsets.contiguous().view(-1)
            scale_o = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-3)

            gs = (STE_multistep.apply(scaling_c.view(-1), Qs)).detach()
            go = (STE_multistep.apply(offsets_c.view(-1, 3 * self.n_offsets).view(-1), Qo)).detach()
            mask_tmp = mask_c.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1)

            EG = self.EG_mix_prob_3 if self.use_3gmm else self.EG_mix_prob_2
            bf = EG.forward(feat_q, *mean_list, *scale_list, *probs_list, Q=Qf)
            bs = self.entropy_gaussian.forward(gs, mean_s, scale_s, Qs)
            bo = self.entropy_gaussian.forward(go, mean_o, scale_o, Qo)
            bo = bo * mask_tmp

            total_feat_bits    += torch.sum(bf).item()
            total_scaling_bits += torch.sum(bs).item()
            total_offsets_bits += torch.sum(bo).item()

        return total_feat_bits, total_scaling_bits, total_offsets_bits

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        if self.use_causal_knn:
            # Two-stage (see compute_total_bits): reuse the same neighbor aggregation for both
            # stages so stage 2 can condition feat's context on scaling/offset via FiLM.
            _anchor_int = torch.round(_anchor / self.voxel_size).long()
            sorted_indices = calculate_morton_order(_anchor_int)
            unsort_indices = torch.argsort(sorted_indices)
            anchor_sorted = _anchor[sorted_indices]
            hash_feats = self.calc_interp_feat(anchor_sorted)
            causal_ctx = self.causal_knn.aggregate_only(hash_feats, anchor_sorted, chunk_size=MAX_batch_size)
            feat_context_no_cond = self.causal_knn.apply_fusion(hash_feats, causal_ctx)[unsort_indices]
        else:
            feat_context_no_cond = self.calc_interp_feat(_anchor)

        # Stage 1: scaling/offset entropy params from the unconditioned context.
        _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj_no_cond, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context_no_cond), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
        # Same floors conduct_encoding() clamps to -- match exactly so the two paths quantize
        # scaling/offset identically.
        Q_scaling = torch.clamp(Q_scaling * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
        Q_offsets = torch.clamp(Q_offsets * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

        # Quantize scaling/offset *before* building stage 2's context, so stage 2 conditions on
        # the same actually-quantized values conduct_encoding() will condition on (and transmit),
        # not on the raw continuous parameters -- otherwise this estimate and the real encoded
        # size diverge whenever AnchorCondNorm's FiLM conditioning is actually live.
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling, self.get_scaling.mean())).detach()
        offsets_full = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1), self._offset.mean())).detach()
        # Zero out pruned (masked-off) offset slots, matching conduct_encoding()'s
        # `offsets[~mask] = 0.0` -- these slots are never trained against any rendering loss, so
        # their raw parameter value is unconstrained garbage; conditioning on it (instead of the 0
        # the decoder will actually see) inflates offsets_full's variance and throws off feat's
        # FiLM context, which is what caused the real estimate/encode gap.
        offsets_full = offsets_full * _mask.repeat(1, 1, 3).to(offsets_full.dtype)

        if self.use_causal_knn:
            hash_feats_cond = self.calc_interp_feat(anchor_sorted, grid_scaling[sorted_indices], offsets_full[sorted_indices])
            feat_context_with_scale_offset = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx)[unsort_indices]
            print(f"[DEBUG2 estimate_final_bits] hash_feats: {hash_feats.mean().item():.4f}/{hash_feats.std().item():.4f} "
                  f"causal_ctx: {causal_ctx.mean().item():.4f}/{causal_ctx.std().item():.4f} "
                  f"grid_scaling: {grid_scaling.mean().item():.4f}/{grid_scaling.std().item():.4f} "
                  f"offsets_full: {offsets_full.mean().item():.4f}/{offsets_full.std().item():.4f} "
                  f"hash_feats_cond: {hash_feats_cond.mean().item():.4f}/{hash_feats_cond.std().item():.4f} "
                  f"feat_ctx_wso: {feat_context_with_scale_offset.mean().item():.4f}/{feat_context_with_scale_offset.std().item():.4f} "
                  f"N_anchor={anchor_sorted.shape[0]}")
        else:
            feat_context_with_scale_offset = self.calc_interp_feat(_anchor, grid_scaling, offsets_full)

        mean, scale, prob, _, _, _, _, _, _, _ = \
            torch.split(self.get_grid_mlp(feat_context_with_scale_offset), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
        # See gaussian_renderer/__init__.py: Q_feat_adj always comes from the unconditioned
        # context, never the FiLM-conditioned one.
        Q_feat_adj = Q_feat_adj_no_cond
        Q_feat = torch.clamp(Q_feat * (1 + torch.tanh(Q_feat_adj)), min=0.0001)
        _feat = (STE_multistep.apply(_feat, Q_feat, self._anchor_feat.mean())).detach()
        print(f"[DEBUG estimate_final_bits] mean: {mean.mean().item():.4f}/{mean.std().item():.4f} "
              f"scale: {scale.mean().item():.4f}/{scale.std().item():.4f} "
              f"prob: {prob.mean().item():.4f}/{prob.std().item():.4f} "
              f"Q_feat: {Q_feat.mean().item():.6f}/{Q_feat.std().item():.6f} min={Q_feat.min().item():.6f} max={Q_feat.max().item():.6f} "
              f"feat: {_feat.mean().item():.4f}/{_feat.std().item():.4f} N={_feat.shape[0]}")
        mean_list, scale_list, probs_list = self.get_feat_mixture(_feat, mean, scale, prob)

        offsets = offsets_full.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        EG = self.EG_mix_prob_3 if self.use_3gmm else self.EG_mix_prob_2
        bit_feat = EG.forward(_feat, *mean_list, *scale_list, *probs_list, Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        if self.use_reno:
            # Real RENO-compressed byte count for the exact anchor set conduct_encoding()
            # will compress, instead of the pre-RENO fixed-bit-per-coordinate placeholder
            # (anchor_round_digits) which has nothing to do with RENO's actual entropy coding
            # and overstates its cost by ~4x.
            from utils.reno_utils import compress_reno
            _anchor_int_est = torch.round(_anchor / self.voxel_size)
            bit_anchor = len(compress_reno(_anchor_int_est, ckpt_path=self.reno_ckpt_path)) * 8
        else:
            bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        # A handful of anchors (out of hundreds of thousands) occasionally land on a numerical
        # cliff edge where the entropy model's predicted mean/scale/prob comes out NaN/Inf --
        # observed on real trained checkpoints, sensitive to the exact quantized MLP weights
        # (which anchors trip it isn't consistent across fp16/fp8/int8/etc, so it isn't simply
        # "coarser quantization = more outliers"). torch.sum propagates a single NaN/Inf to the
        # entire total, making the size estimate useless for the other 99.99%+ of anchors that
        # are fine. Zeroing out just the non-finite contributions before summing is a
        # conservative approximation (a real encoder still has to spend some bits on these
        # anchors via whatever escape mechanism it uses) but keeps the aggregate usable instead
        # of NaN.
        bit_feat = torch.nan_to_num(bit_feat, nan=0.0, posinf=0.0, neginf=0.0).sum().item()
        bit_scaling = torch.nan_to_num(bit_scaling, nan=0.0, posinf=0.0, neginf=0.0).sum().item()
        bit_offsets = torch.nan_to_num(bit_offsets, nan=0.0, posinf=0.0, neginf=0.0).sum().item()
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
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        if self.use_reno:
            from utils.reno_utils import compress_reno
            means_strings = compress_reno(_anchor_int, ckpt_path=self.reno_ckpt_path, verbose=True)
        else:
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

        # Pre-compute causal KNN context for all anchors (already sorted in Morton order).
        # The expensive O(N*K) neighbor search (aggregate_only) is shared between the two encoding
        # stages below; only the cheap per-anchor calc_interp_feat + apply_fusion is redone per
        # chunk for stage 2 so feat's context can condition on the just-encoded scaling/offset via
        # FiLM (AnchorCondNorm) -- mirroring the non-causal_knn branch's two-stage design.
        if self.use_causal_knn:
            _hash_feats_all = self.calc_interp_feat(_anchor)
            causal_ctx_all = self.causal_knn.aggregate_only(_hash_feats_all, _anchor, chunk_size=MAX_batch_size)
            all_feat_context_no_cond = self.causal_knn.apply_fusion(_hash_feats_all, causal_ctx_all)
            print(f"[DEBUG2 conduct_encoding] hash_feats: {_hash_feats_all.mean().item():.4f}/{_hash_feats_all.std().item():.4f} "
                  f"causal_ctx: {causal_ctx_all.mean().item():.4f}/{causal_ctx_all.std().item():.4f} "
                  f"N_anchor={_anchor.shape[0]}")

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        _dbg_mean, _dbg_scale, _dbg_prob, _dbg_qfeat, _dbg_feat = [], [], [], [], []
        _dbg_scaling_cond, _dbg_offsets_cond, _dbg_hfc, _dbg_fctx = [], [], [], []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_slice = _anchor[N_start:N_end]
            N_num = N_end - N_start

            if self.use_causal_knn:
                # Two-stage (see precompute comment above): stage 1 = scaling/offsets from the
                # unconditioned context; stage 2 = feat conditioned on the just-quantized
                # scaling/offset via FiLM (AnchorCondNorm), reusing causal_ctx_all so no extra
                # neighbor search is needed.
                _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    self.forward_grid(all_feat_context_no_cond[N_start:N_end])
                # Q_feat_adj deliberately comes from this unconditioned stage-1 pass, not the
                # FiLM-conditioned stage-2 pass below -- see gaussian_renderer/__init__.py for why.
                Q_feat_adj = Q_feat_adj.contiguous().repeat(1, self.feat_dim)

                Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
                Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
                mean_scaling = mean_scaling.contiguous().view(-1)
                mean_offsets = mean_offsets.contiguous().view(-1)
                scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-3)
                scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-3)
                Q_scaling = torch.clamp(Q_scaling * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
                Q_offsets = torch.clamp(Q_offsets * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

                t_scaling_0 = get_time()
                scaling = _scaling[N_start:N_end].view(-1)
                scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
                torch.cuda.synchronize(); t0 = time.time()
                bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_scaling_list.append(bit_scaling)
                t_scaling += get_time() - t_scaling_0

                t_offset_0 = get_time()
                mask = _mask[N_start:N_end].repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)
                offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)
                offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
                offsets[~mask] = 0.0
                torch.cuda.synchronize(); t0 = time.time()
                bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name, chunk_size=10_0000)
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_offsets_list.append(bit_offsets)
                t_offset += get_time() - t_offset_0

                scaling_for_cond = scaling.view(N_num, 6).detach()
                offsets_for_cond = offsets.view(N_num, self.n_offsets, 3).detach()

                hash_feats_cond = self.calc_interp_feat(anchor_slice, anchor_scale=scaling_for_cond, anchor_offset=offsets_for_cond)
                feat_context = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx_all[N_start:N_end])
                mean, scale, prob, _, _, _, _, _, _, _ = \
                    self.forward_grid(feat_context)
                Q_feat = torch.clamp(Q_feat * (1 + torch.tanh(Q_feat_adj)), min=0.0001)

                _dbg_scaling_cond.append(scaling_for_cond.detach().reshape(-1))
                _dbg_offsets_cond.append(offsets_for_cond.detach().reshape(-1))
                _dbg_hfc.append(hash_feats_cond.detach().reshape(-1))
                _dbg_fctx.append(feat_context.detach().reshape(-1))

                feat = _feat[N_start:N_end]
                feat = STE_multistep.apply(feat, Q_feat, self._anchor_feat.mean())
                torch.cuda.synchronize(); t0 = time.time()

                _dbg_mean.append(mean.detach().reshape(-1))
                _dbg_scale.append(scale.detach().reshape(-1))
                _dbg_prob.append(prob.detach().reshape(-1))
                _dbg_qfeat.append(Q_feat.detach().reshape(-1))
                _dbg_feat.append(feat.detach().reshape(-1))

                t_feature_0 = get_time()
                mean_scale_ctx = torch.cat([mean, scale, prob], dim=-1)
                scale = torch.clamp(scale, min=1e-3)
                bit_feat = 0
                for cc in range(5):
                    mean_list, scale_list, probs_list = self.get_feat_mixture(
                        feat, mean[:, cc*10:cc*10+10], scale[:, cc*10:cc*10+10], prob[:, cc*10:cc*10+10],
                        mean_scale_ctx=mean_scale_ctx, to_dec=cc)
                    feat_tmp = feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    bit_feat += encoder_gaussian_mixed_chunk(
                        feat_tmp,
                        [m.contiguous().view(-1) for m in mean_list],
                        [s.contiguous().view(-1) for s in scale_list],
                        [p.contiguous().view(-1) for p in probs_list],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
                t_feature += get_time() - t_feature_0
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_feat_list.append(bit_feat)

            else:
                # Two-stage: Stage 1 = scaling/offsets (no conditioning), Stage 2 = feat (conditioned on scaling/offset)
                # Stage 1: encode scaling/offsets without conditioning
                feat_context_no_cond = self.calc_interp_feat(anchor_slice)
                _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(self.get_grid_mlp(feat_context_no_cond), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
                # Q_feat_adj deliberately comes from this unconditioned stage-1 pass, not stage 2
                # below -- see gaussian_renderer/__init__.py for why.
                Q_feat_adj = Q_feat_adj.contiguous().repeat(1, self.feat_dim)

                Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
                Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
                mean_scaling = mean_scaling.contiguous().view(-1)
                mean_offsets = mean_offsets.contiguous().view(-1)
                scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-3)
                scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-3)
                Q_scaling = torch.clamp(Q_scaling * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
                Q_offsets = torch.clamp(Q_offsets * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

                t_scaling_0 = get_time()
                scaling = _scaling[N_start:N_end].view(-1)
                scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
                torch.cuda.synchronize(); t0 = time.time()
                bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_scaling_list.append(bit_scaling)
                t_scaling += get_time() - t_scaling_0

                t_offset_0 = get_time()
                mask = _mask[N_start:N_end]  # {0, 1}  # [N_num, K, 1]
                mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
                offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
                offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
                offsets[~mask] = 0.0
                torch.cuda.synchronize(); t0 = time.time()
                bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name, chunk_size=10_0000)
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_offsets_list.append(bit_offsets)
                t_offset += get_time() - t_offset_0

                scaling_for_cond = scaling.view(N_num, 6).detach()
                offsets_for_cond = offsets.view(N_num, self.n_offsets, 3).detach()

                # Stage 2: encode feat conditioned on quantized scaling and offset
                feat_context_with_scale_offset = self.calc_interp_feat(anchor_slice, anchor_scale=scaling_for_cond, anchor_offset=offsets_for_cond)
                mean, scale, prob, _, _, _, _, _, _, _ = \
                    torch.split(self.get_grid_mlp(feat_context_with_scale_offset), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

                Q_feat = torch.clamp(Q_feat * (1 + torch.tanh(Q_feat_adj)), min=0.0001)

                feat = _feat[N_start:N_end]
                feat = STE_multistep.apply(feat, Q_feat, self._anchor_feat.mean())
                torch.cuda.synchronize(); t0 = time.time()

                t_feature_0 = get_time()
                mean_scale_ctx = torch.cat([mean, scale, prob], dim=-1)
                scale_feat = torch.clamp(scale, min=1e-3)
                bit_feat = 0
                for cc in range(5):
                    mean_list, scale_list, probs_list = self.get_feat_mixture(
                        feat, mean[:, cc*10:cc*10+10], scale_feat[:, cc*10:cc*10+10], prob[:, cc*10:cc*10+10],
                        mean_scale_ctx=mean_scale_ctx, to_dec=cc)
                    feat_tmp = feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    bit_feat += encoder_gaussian_mixed_chunk(
                        feat_tmp,
                        [m.contiguous().view(-1) for m in mean_list],
                        [s.contiguous().view(-1) for s in scale_list],
                        [p.contiguous().view(-1) for p in probs_list],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
                t_feature += get_time() - t_feature_0
                torch.cuda.synchronize(); t_codec += time.time() - t0
                bit_feat_list.append(bit_feat)

            torch.cuda.empty_cache()

        bit_anchor = bits_xyz
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        if _dbg_mean:
            _m = torch.cat(_dbg_mean); _s = torch.cat(_dbg_scale); _p = torch.cat(_dbg_prob)
            _q = torch.cat(_dbg_qfeat); _f = torch.cat(_dbg_feat)
            print(f"[DEBUG conduct_encoding] mean: {_m.mean().item():.4f}/{_m.std().item():.4f} "
                  f"scale: {_s.mean().item():.4f}/{_s.std().item():.4f} "
                  f"prob: {_p.mean().item():.4f}/{_p.std().item():.4f} "
                  f"Q_feat: {_q.mean().item():.6f}/{_q.std().item():.6f} min={_q.min().item():.6f} max={_q.max().item():.6f} "
                  f"feat: {_f.mean().item():.4f}/{_f.std().item():.4f} N={_f.shape[0]}")
            _sc = torch.cat(_dbg_scaling_cond); _oc = torch.cat(_dbg_offsets_cond)
            _hfc = torch.cat(_dbg_hfc); _fctx = torch.cat(_dbg_fctx)
            print(f"[DEBUG2 conduct_encoding] scaling_for_cond: {_sc.mean().item():.4f}/{_sc.std().item():.4f} "
                  f"offsets_for_cond: {_oc.mean().item():.4f}/{_oc.std().item():.4f} "
                  f"hash_feats_cond: {_hfc.mean().item():.4f}/{_hfc.std().item():.4f} "
                  f"feat_context: {_fctx.mean().item():.4f}/{_fctx.std().item():.4f}")

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
        data_dict = np.load(npz_path, allow_pickle=True)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        if self.use_reno:
            from utils.reno_utils import decompress_reno
            _anchor_int_dec = decompress_reno(means_strings, ckpt_path=self.reno_ckpt_path, verbose=True).to('cuda')
        else:
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

        # Pre-compute causal KNN context for all decoded anchors (already in Morton order).
        # Two-stage, mirroring conduct_encoding: the expensive neighbor search is shared between
        # stages, only the cheap calc_interp_feat + apply_fusion is redone per chunk for stage 2
        # so feat's context can condition on the just-decoded scaling/offset via FiLM.
        if self.use_causal_knn:
            _hash_feats_all_dec = self.calc_interp_feat(anchor_decoded)
            causal_ctx_all_dec = self.causal_knn.aggregate_only(_hash_feats_all_dec, anchor_decoded, chunk_size=MAX_batch_size)
            all_feat_context_no_cond_dec = self.causal_knn.apply_fusion(_hash_feats_all_dec, causal_ctx_all_dec)

        for s in range(steps):

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_sort = anchor_decoded[N_start:N_end]

            if self.use_causal_knn:
                # Two-stage (see precompute comment above): stage 1 = scaling/offsets from the
                # unconditioned context.
                _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    self.forward_grid(all_feat_context_no_cond_dec[N_start:N_end])
                # Q_feat_adj deliberately comes from this unconditioned stage-1 pass, matching
                # conduct_encoding -- see gaussian_renderer/__init__.py for why.
                Q_feat_adj = Q_feat_adj.contiguous().repeat(1, self.feat_dim)

                Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
                Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
                mean_scaling = mean_scaling.contiguous().view(-1)
                mean_offsets = mean_offsets.contiguous().view(-1)
                scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-3)
                scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-3)
                Q_scaling = torch.clamp(Q_scaling * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
                Q_offsets = torch.clamp(Q_offsets * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

                t_scaling_0 = get_time()
                scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
                scaling_decoded = scaling_decoded.view(N_num, 6)
                scaling_decoded_list.append(scaling_decoded)
                t_scaling += get_time() - t_scaling_0

                t_offset_0 = get_time()
                masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
                offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, chunk_size=10_0000)
                offsets_decoded = torch.zeros_like(mean_offsets)
                offsets_decoded[masks_tmp] = offsets_decoded_tmp
                offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)
                offsets_decoded_list.append(offsets_decoded)
                t_offset += get_time() - t_offset_0

                # Stage 2: feat conditioned on the just-decoded scaling/offset via FiLM
                hash_feats_cond = self.calc_interp_feat(anchor_sort, anchor_scale=scaling_decoded.detach(), anchor_offset=offsets_decoded.detach())
                feat_context = self.causal_knn.apply_fusion(hash_feats_cond, causal_ctx_all_dec[N_start:N_end])
                mean, scale, prob, _, _, _, _, _, _, _ = \
                    self.forward_grid(feat_context)
                Q_feat = torch.clamp(Q_feat * (1 + torch.tanh(Q_feat_adj)), min=0.0001)

                t_feature_0 = get_time()
                feat_decoded = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
                mean_scale_ctx = torch.cat([mean, scale, prob], dim=-1)
                scale = torch.clamp(scale, min=1e-3)
                for cc in range(5):
                    mean_list, scale_list, probs_list = self.get_feat_mixture(
                        feat_decoded, mean[:, cc*10:cc*10+10], scale[:, cc*10:cc*10+10], prob[:, cc*10:cc*10+10],
                        mean_scale_ctx=mean_scale_ctx, to_dec=cc)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                        [m.contiguous().view(-1) for m in mean_list],
                        [s.contiguous().view(-1) for s in scale_list],
                        [p.contiguous().view(-1) for p in probs_list],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
                    feat_decoded_tmp = feat_decoded_tmp.view(N_num, 10)
                    feat_decoded[:, cc*10:cc*10+10] = feat_decoded_tmp
                feat_decoded_list.append(feat_decoded)
                t_feature += get_time() - t_feature_0

            else:
                # Two-stage: Stage 1 = scaling/offsets (no conditioning), Stage 2 = feat (conditioned on scaling/offset)
                # Stage 1: decode scaling/offsets without conditioning
                feat_context_no_cond = self.calc_interp_feat(anchor_sort)
                _, _, _, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(self.get_grid_mlp(feat_context_no_cond), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
                # Q_feat_adj deliberately comes from this unconditioned stage-1 pass, matching
                # conduct_encoding -- see gaussian_renderer/__init__.py for why.
                Q_feat_adj = Q_feat_adj.contiguous().repeat(1, self.feat_dim)

                Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
                Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
                mean_scaling = mean_scaling.contiguous().view(-1)
                mean_offsets = mean_offsets.contiguous().view(-1)
                scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-3)
                scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-3)
                Q_scaling = torch.clamp(Q_scaling * (1 + torch.tanh(Q_scaling_adj)), min=0.00001)
                Q_offsets = torch.clamp(Q_offsets * (1 + torch.tanh(Q_offsets_adj)), min=0.0001)

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

                # Stage 2: decode feat conditioned on decoded scaling and offset
                feat_context_with_scale_offset = self.calc_interp_feat(anchor_sort, anchor_scale=scaling_decoded.detach(), anchor_offset=offsets_decoded.detach())
                mean, scale, prob, _, _, _, _, _, _, _ = \
                    torch.split(self.get_grid_mlp(feat_context_with_scale_offset), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

                Q_feat = torch.clamp(Q_feat * (1 + torch.tanh(Q_feat_adj)), min=0.0001)

                t_feature_0 = get_time()
                feat_decoded = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
                mean_scale_ctx = torch.cat([mean, scale, prob], dim=-1)
                scale_feat = torch.clamp(scale, min=1e-3)
                for cc in range(5):
                    mean_list, scale_list, probs_list = self.get_feat_mixture(
                        feat_decoded, mean[:, cc*10:cc*10+10], scale_feat[:, cc*10:cc*10+10], prob[:, cc*10:cc*10+10],
                        mean_scale_ctx=mean_scale_ctx, to_dec=cc)
                    Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)
                    feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                        [m.contiguous().view(-1) for m in mean_list],
                        [s.contiguous().view(-1) for s in scale_list],
                        [p.contiguous().view(-1) for p in probs_list],
                        Q_feat_tmp,
                        file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
                    feat_decoded_tmp = feat_decoded_tmp.view(N_num, 10)
                    feat_decoded[:, cc*10:cc*10+10] = feat_decoded_tmp
                feat_decoded_list.append(feat_decoded)
                t_feature += get_time() - t_feature_0

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


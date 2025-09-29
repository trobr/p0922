import torch
import tinycudann as tcnn
import torch.nn.functional as F

from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, feat_dim, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim // reduction)
        self.fc2 = nn.Linear(feat_dim // reduction, feat_dim)

    def forward(self, x):  # [N, F]
        w = F.relu(self.fc1(x.mean(dim=0))) 
        w = torch.sigmoid(self.fc2(w))
        return x * w


class PointSelfAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):  # [N, F]
        out, _ = self.attn(x, x, x)
        return out


class LocalAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4, voxel_size=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.voxel_size = voxel_size

    def forward(self, x, coords):
        """
        x: [N, F] 特征
        coords: [N, 3] 原始坐标 (归一化到 [0,1])
        """
        # 把点量化到 voxel
        voxel_idx = torch.floor(coords / self.voxel_size).long()
        voxel_keys = voxel_idx[:,0] * 73856093 ^ voxel_idx[:,1] * 19349663 ^ voxel_idx[:,2] * 83492791  # hash voxel
        unique_voxels = voxel_keys.unique()
        
        outs = torch.zeros_like(x)
        for v in unique_voxels:
            mask = voxel_keys == v
            if mask.sum() < 2:  # 单点跳过
                outs[mask] = x[mask]
                continue
            feat_block = x[mask].unsqueeze(0)  # [1, n, F]
            out_block, _ = self.attn(feat_block, feat_block, feat_block)
            outs[mask] = out_block.squeeze(0)
        return outs


class ChannelSE(nn.Module):
    """Lightweight channel attention (SE-like) operating on [N, F]."""
    def __init__(self, feat_dim, reduction=8):
        super().__init__()
        hidden = max(4, feat_dim // reduction)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, feat_dim)

    def forward(self, x):
        s = x.mean(dim=0, keepdim=True)   # [1, F] global pooling over points
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))    # [1, F]
        return x * s  # broadcast to [N, F]


class InstanceSE(nn.Module):
    """Lightweight channel attention (SE-like) operating on [N, F]."""
    def __init__(self, feat_dim, reduction=8, alpha=0.):
        super().__init__()
        hidden = max(1, feat_dim // reduction)
        self.fc1 = nn.Linear(feat_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, feat_dim, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        g = self.fc2(F.relu(self.fc1(x)))  # [N, F]
        gate = 1.0 + self.alpha * torch.tanh(g)  # [N, F]
        return x * gate  # [N, F] broadcast


class LocalVoxelAttention(nn.Module):
    """
    Voxelized local attention.
    - coords: [N, 3], in world coordinates or normalized coords (controls voxel_size)
    - feat:   [N, F]
    Strategy:
      - voxelize coords by floor(coords / voxel_size)
      - group points by voxel
      - for each voxel group, split into chunks of size <= max_points_per_voxel and apply MultiheadAttention
      - if group size == 1 -> skip (identity)
    """
    def __init__(self, feat_dim, num_heads=4, voxel_size=0.02, max_points_per_voxel=128, attn_dropout=0.0):
        super().__init__()
        self.voxel_size = voxel_size
        self.max_points = max_points_per_voxel
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True, dropout=attn_dropout)

    @staticmethod
    def _voxel_hash(vx, vy, vz):
        return (vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791)

    def forward(self, feat, coords):
        """
        feat:  [N, F]
        coords: [N, 3]
        returns: [N, F]
        """
        device = feat.device
        N, F = feat.shape

        scaled = torch.floor(coords / self.voxel_size).to(torch.int64)  # [N,3]
        vx, vy, vz = scaled[:, 0], scaled[:, 1], scaled[:, 2]
        keys = self._voxel_hash(vx.to(device), vy.to(device), vz.to(device))

        unique_keys, inverse_idx = torch.unique(keys, return_inverse=True)
        out = torch.empty_like(feat)

        for vk in range(unique_keys.shape[0]):
            mask = (inverse_idx == vk)
            idxs = torch.nonzero(mask, as_tuple=False).view(-1)
            m = idxs.numel()
            if m <= 1:
                # trivial case: identity
                out[idxs] = feat[idxs]
                continue

            start = 0
            while start < m:
                end = min(m, start + self.max_points)
                chunk_idxs = idxs[start:end]
                qkv = feat[chunk_idxs].unsqueeze(0)  # [1, L, F]
                attn_out, _ = self.attn(qkv, qkv, qkv)
                out[chunk_idxs] = attn_out.squeeze(0)
                start = end

        return out


class HashEncoderAttn(nn.Module):
    """
    HashGrid encoding + per-feature gain + (Channel SE) + LocalVoxelAttention + Linear + small Residual MLP.

    Interface:
      - input:  x [P, N, 3]
      - output: y [P, N, C]
    """
    def __init__(self,
                 num_channels,
                 num_players=1,
                 hash_config=None,
                 voxel_size=0.02,
                 max_points_per_voxel=128,
                 attn_heads=4,
                 se_reduction=8):
        super().__init__()
        self.num_players = num_players
        self.num_channels = num_channels

        default_hash = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 17,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        }
        hash_config = hash_config or default_hash

        # per-player modules (keeps interface identical)
        self.encoders = nn.ModuleList()
        self.gains = nn.ParameterList()
        self.channel_se = nn.ModuleList()
        self.local_attn = nn.ModuleList()
        self.linear_heads = nn.ModuleList()
        self.residual_mlps = nn.ModuleList()

        for _ in range(num_players):
            enc = tcnn.Encoding(n_input_dims=3, encoding_config=hash_config)
            feat_dim = int(enc.n_output_dims)

            self.encoders.append(enc)
            self.gains.append(nn.Parameter(torch.ones(feat_dim)))
            self.channel_se.append(ChannelSE(feat_dim, reduction=se_reduction))
            self.local_attn.append(LocalVoxelAttention(feat_dim,
                                                       num_heads=attn_heads,
                                                       voxel_size=voxel_size,
                                                       max_points_per_voxel=max_points_per_voxel))
            # heads
            self.linear_heads.append(nn.Linear(feat_dim, num_channels))
            # small residual mlp: narrow for efficiency
            hidden = min(64, feat_dim)
            self.residual_mlps.append(nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_channels)
            ))

    def forward(self, x, coords_world=None):
        """
        x: [P, N, 3] => locations (these go into tcnn encoding)
        coords_world: optional [P, N, 3] world coords for voxelization/attention.
                      If None, uses x as coords (assumes x in consistent space).
        returns: [P, N, C]
        """
        P = self.num_players
        assert x.shape[0] == P, f"expected first dim == num_players ({P})"
        outs = []
        for pid in range(P):
            coords = x[pid] if coords_world is None else coords_world[pid]  # [N,3]
            device = coords.device

            # 1) HashGrid encoding -> feat [N, F]
            feat = self.encoders[pid](x[pid].to(device)).float()  # ensure device match

            # 2) per-feature gain (broadcast)
            feat = feat * self.gains[pid].to(feat.dtype)

            # 3) Channel attention (SE-like)
            feat = self.channel_se[pid](feat)

            # 4) Local voxel attention (operates on feat, coords)
            feat = self.local_attn[pid](feat, coords)

            # 5) Linear head + residual MLP
            y = self.linear_heads[pid](feat) + self.residual_mlps[pid](feat)

            outs.append(y)

        return torch.stack(outs)  # [P, N, C]


class HashEncoderResidualAttn(nn.Module):
    """
    HashGrid encoding + per-feature gain + Channel Attention + Linear + small Residual MLP
    Input:  x [P, N, 3]
    Output: y [P, N, C]
    Fully parallel, high efficiency
    """
    def __init__(self, num_channels, num_players=1, hash_config=None, se_reduction=8):
        super().__init__()
        self.num_players = num_players
        self.num_channels = num_channels

        default_hash = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 17,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        }
        hash_config = hash_config or default_hash

        self.encoders = nn.ModuleList()
        self.gains = nn.ParameterList()
        self.channel_se = nn.ModuleList()
        self.linear_heads = nn.ModuleList()
        self.residual_mlps = nn.ModuleList()

        for _ in range(num_players):
            enc = tcnn.Encoding(n_input_dims=3, encoding_config=hash_config)
            feat_dim = int(enc.n_output_dims)

            self.encoders.append(enc)
            self.gains.append(nn.Parameter(torch.ones(feat_dim)))
            self.channel_se.append(ChannelSE(feat_dim, reduction=se_reduction))
            self.linear_heads.append(nn.Linear(feat_dim, num_channels))

            hidden = min(64, feat_dim)
            self.residual_mlps.append(nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_channels)
            ))

    def forward(self, x):
        """
        x: [P, N, 3]
        returns: [P, N, C]
        """
        P = self.num_players
        assert x.shape[0] == P, f"expected first dim == num_players ({P})"
        outs = []
        for pid in range(P):
            # HashGrid encoding
            feat = self.encoders[pid](x[pid])   # [N, F]

            # per-feature gain
            feat = feat * self.gains[pid]

            # channel attention (SE)
            feat = self.channel_se[pid](feat)

            # Linear head + Residual MLP
            y = self.linear_heads[pid](feat) + self.residual_mlps[pid](feat)

            outs.append(y)

        return torch.stack(outs)  # [P, N, C]


class HashEncoderInstAttn(nn.Module):
    """
    原 HashEncoder + Channel SE Attention
    """
    def __init__(self, num_channels, num_players=1, se_reduction=8):
        super().__init__()
        self.networks = nn.ModuleList()
        self.channel_se = nn.ModuleList()
        self.num_players = num_players

        for i in range(num_players):
            net = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=num_channels,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 17,
                    "base_resolution": 4,
                    "per_level_scale": 1.5,
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
            self.networks.append(net)
            self.channel_se.append(InstanceSE(num_channels, reduction=se_reduction))

    def forward(self, x):
        """
        x: [P, N, 3]
        returns: [P, N, C]
        """
        outs = []
        for i in range(self.num_players):
            feat = self.networks[i](x[i]).float()  # [N, C]
            feat = self.channel_se[i](feat)        # SE Attention
            outs.append(feat)
        return torch.stack(outs).float()           # [P, N, C]


class _HashEncoderSeAttn(nn.Module):
    def __init__(self, num_channels, num_players=1):
        super().__init__()
        self.encoder = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=num_channels,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 17,
                    "base_resolution": 4,
                    "per_level_scale": 1.5,
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                })
        self.se = ChannelSE(num_channels, reduction=8)
        # self.gamma = nn.Parameter(torch.ones(1, num_channels), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros(1, num_channels), requires_grad=True)
        # self.mlp = nn.Linear(num_channels, num_channels)


    def forward(self, x):
        feat = self.encoder(x).float()
        return self.se(feat)


class HashEncoderSeAttn(nn.Module):
    def __init__(self, num_channels, num_players=1):
        super().__init__()
        self.networks = []
        self.num_players = num_players
        for i in range(num_players):
            self.networks.append(
                _HashEncoderSeAttn(num_channels, num_players)
            )
        self.networks = nn.ModuleList(self.networks)

    def forward(self, x):
        self.outputs = []
        for i in range(self.num_players):
            self.outputs.append(self.networks[i](x[i]))
        return torch.stack(self.outputs).float()

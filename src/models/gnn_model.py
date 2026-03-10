"""
GNN Traffic Prediction Models
==============================
Graph Neural Network architectures for spatio-temporal traffic flow prediction.
Includes STGCN, DCRNN, GraphSAGE-LSTM, and GATv2 variants.

Research Contributions:
  - Improved traffic prediction accuracy via spatio-temporal graph learning
  - Smart transportation planning for autonomous navigation in smart cities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, ChebConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Utility Modules
# ---------------------------------------------------------------------------

class TemporalConvBlock(nn.Module):
    """1-D gated temporal convolution block (used in STGCN)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_gate = nn.Conv2d(in_channels, 2 * out_channels,
                                   kernel_size=(1, kernel_size), padding=(0, pad))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_gate(x)
        p, q = out.chunk(2, dim=1)
        out = p * torch.sigmoid(q)
        out = self.bn(out)
        return self.dropout(out)


class SpatialGraphConvBlock(nn.Module):
    """Chebyshev graph convolution spatial block."""

    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.cheb = ChebConv(in_channels, out_channels, K=K)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        out = self.cheb(x, edge_index, edge_weight)
        return F.relu(self.ln(out))


# ---------------------------------------------------------------------------
# Model 1: STGCN
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):
    """Temporal -> Spatial -> Temporal sandwich block."""

    def __init__(self, in_channels, spatial_channels, out_channels, K=3):
        super().__init__()
        self.temp1 = TemporalConvBlock(in_channels, spatial_channels)
        self.spatial = SpatialGraphConvBlock(spatial_channels, spatial_channels, K)
        self.temp2 = TemporalConvBlock(spatial_channels, out_channels)
        self.residual = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x, edge_index, edge_weight=None):
        B, C, N, T = x.shape
        res = self.residual(x)
        out = self.temp1(x)
        C_s = out.shape[1]
        out_s = out.permute(0, 3, 2, 1).reshape(B * T * N, C_s)
        out_s = self.spatial(out_s, edge_index, edge_weight)
        out = out_s.reshape(B, T, N, C_s).permute(0, 3, 2, 1)
        return self.temp2(out) + res


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Yu et al., IJCAI 2018.
    """

    def __init__(self, num_nodes, in_channels, hidden_channels,
                 out_channels, T_in, T_out, num_blocks=2, K=3):
        super().__init__()
        self.T_out = T_out
        blocks, c = [], in_channels
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(c, hidden_channels, hidden_channels, K))
            c = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.ReLU(),
            nn.Conv2d(hidden_channels, T_out, 1))

    def forward(self, x, edge_index, edge_weight=None):
        """x: (B, T_in, N, C) -> (B, T_out, N)"""
        x = x.permute(0, 3, 2, 1)
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)
        return self.output_conv(x).mean(dim=-1)


# ---------------------------------------------------------------------------
# Model 2: DCRNN
# ---------------------------------------------------------------------------

class DiffusionConv(nn.Module):
    """Bidirectional random-walk diffusion convolution."""

    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.empty(2 * K + 1, in_channels, out_channels))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, adj):
        B, N, C = x.shape
        outputs, Pk = [], torch.eye(N, device=x.device)
        for _ in range(self.K + 1):
            outputs.append(torch.einsum('bni,ij->bnj', x, Pk))
            Pk = Pk @ adj
        Pk, adj_t = torch.eye(N, device=x.device), adj.T
        for _ in range(1, self.K + 1):
            Pk = Pk @ adj_t
            outputs.append(torch.einsum('bni,ij->bnj', x, Pk))
        out = torch.stack(outputs, dim=1)
        return torch.einsum('bknc,kco->bno', out, self.theta)


class DCRNNCell(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, K=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        total_in = in_channels + hidden_channels
        self.conv_z = DiffusionConv(total_in, hidden_channels, K)
        self.conv_r = DiffusionConv(total_in, hidden_channels, K)
        self.conv_c = DiffusionConv(total_in, hidden_channels, K)

    def forward(self, x, h, adj):
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.conv_z(xh, adj))
        r = torch.sigmoid(self.conv_r(xh, adj))
        c = torch.tanh(self.conv_c(torch.cat([x, r * h], -1), adj))
        return z * h + (1 - z) * c


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network.
    Li et al., ICLR 2018.
    """

    def __init__(self, num_nodes, in_channels, hidden_channels,
                 out_channels, T_in, T_out, num_layers=2, K=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.T_out = T_out
        self.num_layers = num_layers
        self.encoder = nn.ModuleList(
            [DCRNNCell(num_nodes, in_channels if i == 0 else hidden_channels,
                       hidden_channels, K) for i in range(num_layers)])
        self.decoder = nn.ModuleList(
            [DCRNNCell(num_nodes, out_channels if i == 0 else hidden_channels,
                       hidden_channels, K) for i in range(num_layers)])
        self.proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        """x: (B, T_in, N, C_in), adj: (N, N) -> (B, T_out, N, C_out)"""
        B, T, N, C = x.shape
        h = [torch.zeros(B, N, self.hidden_channels, device=x.device)
             for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.encoder):
                h[l] = cell(inp, h[l], adj)
                inp = h[l]
        outputs, inp = [], torch.zeros(B, N, 1, device=x.device)
        for _ in range(self.T_out):
            for l, cell in enumerate(self.decoder):
                h[l] = cell(inp, h[l], adj)
                inp = h[l]
            pred = self.proj(inp)
            outputs.append(pred)
            inp = pred
        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Model 3: GATv2-LSTM
# ---------------------------------------------------------------------------

class GATv2LSTM(nn.Module):
    """
    Graph Attention Network v2 (spatial) + BiLSTM (temporal).
    Captures dynamic sensor importance with attention weights.
    """

    def __init__(self, num_nodes, in_channels, gat_hidden, lstm_hidden,
                 out_channels, T_in, T_out, num_heads=4, dropout=0.2):
        super().__init__()
        self.T_out, self.out_channels = T_out, out_channels
        self.gat1 = GATv2Conv(in_channels, gat_hidden, heads=num_heads,
                              dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(gat_hidden * num_heads, gat_hidden,
                              heads=1, dropout=dropout, concat=False)
        self.ln = nn.LayerNorm(gat_hidden)
        self.lstm = nn.LSTM(gat_hidden, lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lstm_hidden, T_out * out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        """x: (B, T_in, N, C_in) -> (B, T_out, N, C_out)"""
        B, T, N, C = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t].reshape(B * N, C)
            s = F.elu(self.gat1(xt, edge_index))
            s = self.ln(self.gat2(s, edge_index))
            feats.append(s.reshape(B, N, -1))
        feats = torch.stack(feats, dim=2).reshape(B * N, T, -1)
        ctx = self.lstm(feats)[0][:, -1]
        out = self.head(ctx).reshape(B, N, self.T_out, self.out_channels)
        return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Model 4: GraphSAGE Baseline
# ---------------------------------------------------------------------------

class GraphSAGETrafficPredictor(nn.Module):
    """Lightweight GraphSAGE baseline — flattens time window as features."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 T_in, T_out, num_layers=3, dropout=0.2):
        super().__init__()
        self.T_in, self.T_out, self.out_channels = T_in, T_out, out_channels
        flat = in_channels * T_in
        self.convs = nn.ModuleList([SAGEConv(flat if i == 0 else hidden_channels,
                                             hidden_channels)
                                    for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)
                                  for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_channels, T_out * out_channels)

    def forward(self, x, edge_index):
        """x: (B, T_in, N, C) -> (B, T_out, N, C_out)"""
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        outs = []
        for b in range(B):
            h = x[b]
            for conv, bn in zip(self.convs, self.bns):
                h = self.drop(F.relu(bn(conv(h, edge_index))))
            outs.append(self.head(h).reshape(N, self.T_out, self.out_channels))
        return torch.stack(outs, 0).permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "stgcn": STGCN,
    "dcrnn": DCRNN,
    "gatv2_lstm": GATv2LSTM,
    "graphsage": GraphSAGETrafficPredictor,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


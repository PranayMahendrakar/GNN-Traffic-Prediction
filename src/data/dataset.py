"""
Traffic Dataset Loaders
========================
Utilities for loading and preprocessing real-world traffic datasets:
  - METR-LA  (Los Angeles loop detector data, 207 sensors)
  - PEMS-BAY (San Francisco Bay Area, 325 sensors)
  - PeMS-D7  (District 7, ~2000 sensors)

Data format expected: H5 or NPZ files with shape (T, N, F)
  T = timesteps, N = sensor nodes, F = features (speed, flow, occupancy)
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import scipy.sparse as sp
from typing import Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Graph Construction Utilities
# ---------------------------------------------------------------------------

def build_sensor_graph(dist_file: str, sigma2: float = 0.1,
                       epsilon: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build adjacency from a pairwise distance CSV.

    Args:
        dist_file: path to CSV with columns [from, to, cost]
        sigma2   : Gaussian kernel variance
        epsilon  : sparsity threshold

    Returns:
        edge_index : (2, E) LongTensor
        edge_weight: (E,) FloatTensor
    """
    df = pd.read_csv(dist_file)
    # Build sensor id -> index mapping
    sensor_ids = sorted(set(df['from'].tolist() + df['to'].tolist()))
    id2idx = {s: i for i, s in enumerate(sensor_ids)}
    N = len(sensor_ids)

    W = np.zeros((N, N), dtype=np.float32)
    for _, row in df.iterrows():
        i, j = id2idx[row['from']], id2idx[row['to']]
        W[i, j] = np.exp(-row['cost'] ** 2 / sigma2)
    W[W < epsilon] = 0.0
    np.fill_diagonal(W, 0.0)

    sp_W = sp.coo_matrix(W)
    edge_index = torch.tensor(np.vstack([sp_W.row, sp_W.col]), dtype=torch.long)
    edge_weight = torch.tensor(sp_W.data, dtype=torch.float)
    return edge_index, edge_weight


def build_knn_graph(coords: np.ndarray, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a k-nearest-neighbour graph from sensor GPS coordinates.

    Args:
        coords: (N, 2) array of (latitude, longitude)
        k     : number of nearest neighbours

    Returns:
        edge_index : (2, E)
        edge_weight: (E,)
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree',
                            metric='haversine').fit(np.radians(coords))
    distances, indices = nbrs.kneighbors(np.radians(coords))

    rows, cols, weights = [], [], []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for d, j in zip(dist_row[1:], idx_row[1:]):
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
            weights.extend([float(np.exp(-d)), float(np.exp(-d))])

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def normalise_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
    """Return row-normalised dense adjacency matrix (N, N)."""
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = edge_weight
    D_inv = torch.diag(1.0 / (A.sum(dim=1).clamp(min=1e-8)))
    return D_inv @ A


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class StandardScaler:
    """Z-score normaliser applied per feature channel."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> 'StandardScaler':
        """data: (T, N, F)"""
        self.mean = data.mean(axis=(0, 1), keepdims=True)   # (1, 1, F)
        self.std = data.std(axis=(0, 1), keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def inverse_transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return t * std + mean


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    Sliding-window dataset for traffic forecasting.

    Args:
        data      : (T, N, F) numpy array — normalised traffic readings
        T_in      : number of historical timesteps (input window)
        T_out     : number of future timesteps (prediction horizon)
        step      : sliding window step (default 1)
    """

    def __init__(self, data: np.ndarray, T_in: int = 12, T_out: int = 12,
                 step: int = 1):
        self.data = torch.from_numpy(data).float()   # (T, N, F)
        self.T_in = T_in
        self.T_out = T_out
        self.indices = list(range(0, len(data) - T_in - T_out + 1, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.data[start: start + self.T_in]           # (T_in, N, F)
        y = self.data[start + self.T_in: start + self.T_in + self.T_out]  # (T_out, N, F)
        return x, y


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def load_metr_la(data_dir: str, T_in: int = 12, T_out: int = 12,
                 val_ratio: float = 0.1, test_ratio: float = 0.2,
                 batch_size: int = 32) -> Dict:
    """
    Load METR-LA dataset.

    Expected files in data_dir:
        metr_la.h5       — traffic speed matrix (T, N)
        adj_mx_metr.csv  — sensor distance CSV [from, to, cost]

    Returns dict with keys: train/val/test DataLoaders, scaler, edge_index,
    edge_weight, adj, num_nodes.
    """
    h5_path = os.path.join(data_dir, 'metr_la.h5')
    adj_path = os.path.join(data_dir, 'adj_mx_metr.csv')

    with h5py.File(h5_path, 'r') as f:
        speed = f['df']['block0_values'][:]   # (T, N)
    data = speed[:, :, np.newaxis]            # (T, N, 1)

    return _split_and_load(data, adj_path, T_in, T_out,
                           val_ratio, test_ratio, batch_size)


def load_pems_bay(data_dir: str, T_in: int = 12, T_out: int = 12,
                  val_ratio: float = 0.1, test_ratio: float = 0.2,
                  batch_size: int = 32) -> Dict:
    """Load PEMS-BAY dataset (same file layout as METR-LA)."""
    h5_path = os.path.join(data_dir, 'pems_bay.h5')
    adj_path = os.path.join(data_dir, 'adj_mx_bay.csv')

    with h5py.File(h5_path, 'r') as f:
        speed = f['df']['block0_values'][:]
    data = speed[:, :, np.newaxis]

    return _split_and_load(data, adj_path, T_in, T_out,
                           val_ratio, test_ratio, batch_size)


def load_npz_dataset(npz_path: str, adj_path: str,
                     T_in: int = 12, T_out: int = 12,
                     val_ratio: float = 0.1, test_ratio: float = 0.2,
                     batch_size: int = 32) -> Dict:
    """
    Generic NPZ loader for PeMS and custom datasets.

    NPZ must contain key 'data' with shape (T, N, F).
    """
    raw = np.load(npz_path)
    data = raw['data'].astype(np.float32)    # (T, N, F)
    return _split_and_load(data, adj_path, T_in, T_out,
                           val_ratio, test_ratio, batch_size)


def _split_and_load(data: np.ndarray, adj_path: str,
                    T_in: int, T_out: int,
                    val_ratio: float, test_ratio: float,
                    batch_size: int) -> Dict:
    T = len(data)
    n_test = int(T * test_ratio)
    n_val = int(T * val_ratio)
    n_train = T - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train: n_train + n_val]
    test_data = data[n_train + n_val:]

    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    train_ds = TrafficDataset(train_data, T_in, T_out)
    val_ds = TrafficDataset(val_data, T_in, T_out)
    test_ds = TrafficDataset(test_data, T_in, T_out)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    edge_index, edge_weight = build_sensor_graph(adj_path)
    N = data.shape[1]
    adj = normalise_adj(edge_index, edge_weight, N)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'scaler': scaler,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'adj': adj,
        'num_nodes': N,
        'num_features': data.shape[2],
    }

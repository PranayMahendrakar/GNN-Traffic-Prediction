"""
Training Script for GNN Traffic Prediction
============================================
Supports STGCN, DCRNN, GATv2-LSTM, and GraphSAGE models.
Includes early stopping, learning-rate scheduling, and TensorBoard logging.

Usage:
    python src/train.py --model stgcn --dataset metr_la \
                        --data_dir ./data --epochs 100
"""

import os
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from models.gnn_model import build_model
from data.dataset import load_metr_la, load_pems_bay, load_npz_dataset
from utils.metrics import masked_mae, masked_rmse, masked_mape

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Traffic Prediction Trainer')

    # Dataset
    parser.add_argument('--dataset', type=str, default='metr_la',
                        choices=['metr_la', 'pems_bay', 'custom'],
                        help='Traffic dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory containing dataset files')
    parser.add_argument('--npz_path', type=str, default=None,
                        help='Path to NPZ file (required for custom dataset)')
    parser.add_argument('--adj_path', type=str, default=None,
                        help='Path to adjacency CSV (required for custom dataset)')

    # Sequence lengths
    parser.add_argument('--T_in', type=int, default=12,
                        help='Input sequence length (timesteps)')
    parser.add_argument('--T_out', type=int, default=12,
                        help='Output prediction horizon (timesteps)')

    # Model
    parser.add_argument('--model', type=str, default='stgcn',
                        choices=['stgcn', 'dcrnn', 'gatv2_lstm', 'graphsage'],
                        help='GNN model architecture')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden channel dimension')
    parser.add_argument('--K', type=int, default=3,
                        help='Chebyshev polynomial order / diffusion steps')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (GATv2-LSTM only)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of ST-Conv blocks (STGCN only)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'])

    # Misc
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu / cuda / auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def load_dataset(args):
    if args.dataset == 'metr_la':
        return load_metr_la(args.data_dir, args.T_in, args.T_out,
                            batch_size=args.batch_size)
    elif args.dataset == 'pems_bay':
        return load_pems_bay(args.data_dir, args.T_in, args.T_out,
                             batch_size=args.batch_size)
    elif args.dataset == 'custom':
        assert args.npz_path and args.adj_path, \
            'Provide --npz_path and --adj_path for custom dataset'
        return load_npz_dataset(args.npz_path, args.adj_path,
                                args.T_in, args.T_out,
                                batch_size=args.batch_size)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def build_model_from_args(args, num_nodes: int, num_features: int) -> nn.Module:
    common = dict(num_nodes=num_nodes, in_channels=num_features,
                  hidden_channels=args.hidden, out_channels=1,
                  T_in=args.T_in, T_out=args.T_out)
    if args.model == 'stgcn':
        return build_model('stgcn', **common, num_blocks=args.num_blocks, K=args.K)
    elif args.model == 'dcrnn':
        return build_model('dcrnn', **common, K=args.K)
    elif args.model == 'gatv2_lstm':
        return build_model('gatv2_lstm',
                           num_nodes=num_nodes, in_channels=num_features,
                           gat_hidden=args.hidden, lstm_hidden=args.hidden,
                           out_channels=1, T_in=args.T_in, T_out=args.T_out,
                           num_heads=args.num_heads, dropout=args.dropout)
    elif args.model == 'graphsage':
        return build_model('graphsage',
                           in_channels=num_features,
                           hidden_channels=args.hidden,
                           out_channels=1, T_in=args.T_in, T_out=args.T_out,
                           dropout=args.dropout)
    else:
        raise ValueError(f'Unknown model: {args.model}')


# ---------------------------------------------------------------------------
# One-epoch functions
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device,
                edge_index, edge_weight, adj, clip_grad, model_name):
    model.train()
    total_loss = 0.0
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    adj = adj.to(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)          # (B, T, N, F)
        optimiser.zero_grad()

        if model_name == 'dcrnn':
            pred = model(x, adj)                    # (B, T_out, N, 1)
        else:
            pred = model(x, edge_index, edge_weight)

        # pred: (B, T_out, N) or (B, T_out, N, 1)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
        target = y[..., 0]                          # (B, T_out, N)

        loss = criterion(pred, target)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, scaler, device,
               edge_index, edge_weight, adj, model_name):
    model.eval()
    preds_all, targets_all = [], []
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    adj = adj.to(device)

    for x, y in loader:
        x = x.to(device)
        if model_name == 'dcrnn':
            pred = model(x, adj)
        else:
            pred = model(x, edge_index, edge_weight)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
        preds_all.append(pred.cpu())
        targets_all.append(y[..., 0])

    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(targets_all, dim=0)

    # Inverse transform
    preds_np = scaler.inverse_transform_tensor(preds.unsqueeze(-1)).squeeze(-1).numpy()
    tgt_np = scaler.inverse_transform_tensor(targets.unsqueeze(-1)).squeeze(-1).numpy()

    mae = masked_mae(preds_np, tgt_np)
    rmse = masked_rmse(preds_np, tgt_np)
    mape = masked_mape(preds_np, tgt_np)
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f'Using device: {device}')

    # Directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    run_name = f'{args.model}_{args.dataset}_{int(time.time())}'
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))

    # Data
    logger.info('Loading dataset...')
    ds = load_dataset(args)
    train_loader = ds['train']
    val_loader = ds['val']
    test_loader = ds['test']
    scaler = ds['scaler']
    edge_index = ds['edge_index']
    edge_weight = ds['edge_weight']
    adj = ds['adj']
    num_nodes = ds['num_nodes']
    num_features = ds['num_features']
    logger.info(f'Nodes: {num_nodes} | Features: {num_features}')

    # Model
    model = build_model_from_args(args, num_nodes, num_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model: {args.model} | Parameters: {n_params:,}')

    # Resume
    start_epoch = 0
    best_val_mae = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_mae = ckpt.get('best_val_mae', float('inf'))
        logger.info(f'Resumed from {args.resume} (epoch {start_epoch})')

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimiser, patience=5, factor=0.5, min_lr=1e-5)
    else:
        scheduler = None

    # Training loop
    patience_counter = 0
    ckpt_path = os.path.join(args.ckpt_dir, f'{run_name}_best.pt')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, criterion,
                                 device, edge_index, edge_weight, adj,
                                 args.clip_grad, args.model)
        val_mae, val_rmse, val_mape = eval_epoch(model, val_loader, scaler,
                                                  device, edge_index, edge_weight,
                                                  adj, args.model)
        elapsed = time.time() - t0

        # LR scheduling
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_mae)
            else:
                scheduler.step()
        current_lr = optimiser.param_groups[0]['lr']

        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('MAPE/val', val_mape, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        logger.info(
            f'Epoch {epoch+1:3d}/{args.epochs} | '
            f'TrainLoss={train_loss:.4f} | '
            f'ValMAE={val_mae:.4f} | ValRMSE={val_rmse:.4f} | '
            f'ValMAPE={val_mape:.2f}% | LR={current_lr:.2e} | {elapsed:.1f}s'
        )

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({'epoch': epoch + 1, 'model': model.state_dict(),
                        'optimiser': optimiser.state_dict(),
                        'best_val_mae': best_val_mae,
                        'args': vars(args)}, ckpt_path)
            logger.info(f'  -> Saved best model (MAE={best_val_mae:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    # Test evaluation
    logger.info('Loading best model for test evaluation...')
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt['model'])
    test_mae, test_rmse, test_mape = eval_epoch(model, test_loader, scaler,
                                                  device, edge_index, edge_weight,
                                                  adj, args.model)
    logger.info(
        f'TEST RESULTS | MAE={test_mae:.4f} | '
        f'RMSE={test_rmse:.4f} | MAPE={test_mape:.2f}%'
    )
    writer.add_hparams(vars(args),
                       {'test/MAE': test_mae, 'test/RMSE': test_rmse,
                        'test/MAPE': test_mape})
    writer.close()


if __name__ == '__main__':
    main()

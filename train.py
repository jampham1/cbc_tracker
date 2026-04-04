"""
Phase 4 — Part 3: Training Entry Point
========================================
Run this file to train the model end-to-end.

Usage:
    python train.py

Expected output (first few epochs):
    Epoch   Train loss   Val loss     Val AUROC   Val sens   Val spec   LR
    ──────────────────────────────────────────────────────────────────────
    1       0.6821       0.6543       0.6812      0.5231     0.7043     1.00e-03
    2       0.5934       0.5821       0.7234      0.6012     0.7421     1.00e-03
    ...
"""

import torch
import numpy as np
import random
from pathlib import Path


# ── Reproducibility ────────────────────────────────────────────────────────────
# Setting seeds ensures you get the same results each run.
# Important for debugging — if results change every run, you can't
# tell whether a code change helped or you just got lucky.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ─────────────────────────────────────────────────────────────────────
# MPS = Apple Silicon GPU, CUDA = NVIDIA GPU, CPU = fallback
device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f'Using device: {device}')

# ─────────────────────────────────────────────────────────────────────────────
# Config — all hyperparameters in one place
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    'csv_path':        'data/cbc_synthetic.csv',
    'output_dir':      'outputs',
    'window_size':     15,
    'stride':          3,
    'batch_size':      32,

    # Model
    'hidden_channels': 64,
    'n_blocks':        4,
    'kernel_size':     3,
    'dropout':         0.2,

    # Training
    'n_epochs':        50,
    'lr':              1e-3,
    'weight_decay':    1e-4,
    'patience':        10,          # early stopping patience
    'ckpt_path':       'outputs/checkpoints/best_model.pt',
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Preprocessing ────────────────────────────────────────────────
    print('\n── Phase 2: Preprocessing ──────────────────────────────')
    from phase2_preprocessing import run_preprocessing

    data = run_preprocessing(
        csv_path    = CONFIG['csv_path'],
        output_dir  = CONFIG['output_dir'],
        window_size = CONFIG['window_size'],
        stride      = CONFIG['stride'],
        batch_size  = CONFIG['batch_size'],
    )

    # ── Phase 3: Model ────────────────────────────────────────────────────────
    print('\n── Phase 3: Model ──────────────────────────────────────')
    from phase3_part4_full_model import CBCAnomalyTCN

    model = CBCAnomalyTCN(
        n_seq_features    = data['n_seq_features'],
        n_static_features = data['n_static_features'],
        hidden_channels   = CONFIG['hidden_channels'],
        n_blocks          = CONFIG['n_blocks'],
        kernel_size       = CONFIG['kernel_size'],
        dropout           = CONFIG['dropout'],
    )

    counts = model.count_parameters()
    print(f"Model parameters:")
    for name, n in counts.items():
        print(f"  {name:<16} {n:>8,}")

    # ── Phase 4: Training ─────────────────────────────────────────────────────
    print('\n── Phase 4: Training ───────────────────────────────────')
    from phase4_part2_training_loop import run_training

    history = run_training(
        model         = model,
        train_loader  = data['train_loader'],
        val_loader    = data['val_loader'],
        train_windows = data['train_windows'],
        device        = device,
        n_epochs      = CONFIG['n_epochs'],
        lr            = CONFIG['lr'],
        weight_decay  = CONFIG['weight_decay'],
        patience      = CONFIG['patience'],
        ckpt_path     = CONFIG['ckpt_path'],
    )

    # ── Save history ──────────────────────────────────────────────────────────
    import json
    history_path = Path(CONFIG['output_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nTraining history saved → {history_path}')
    print('\nNext step: run phase5_evaluate.py to assess model performance.')


if __name__ == '__main__':
    main()

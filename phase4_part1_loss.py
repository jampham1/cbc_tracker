"""
Phase 4 — Part 1: Loss Function + Class Imbalance
===================================================
Defines the weighted binary cross-entropy loss used to train the model.

The problem:
    ~30% of timesteps are anomalous, ~70% are normal.
    An untrained model that always outputs 0 ("normal") achieves
    70% accuracy without learning anything useful.
    This is called the class imbalance problem.

The fix — pos_weight:
    BCEWithLogitsLoss accepts a `pos_weight` argument.
    Setting pos_weight = n_negatives / n_positives = 70/30 ≈ 2.33
    tells the loss to penalise missing an anomaly 2.33x more than
    a false alarm. The model is forced to take anomalies seriously.

Why BCEWithLogitsLoss over BCELoss?
    BCELoss expects sigmoid(logits) as input. Computing sigmoid then
    log(sigmoid(x)) has catastrophic cancellation for large |x|.
    BCEWithLogitsLoss uses the log-sum-exp trick internally:
        loss = max(x, 0) - x*y + log(1 + exp(-|x|))
    which is numerically stable across the full range of logits.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_pos_weight(windows: list, device: torch.device) -> torch.Tensor:
    """
    Compute the positive class weight from the training windows.

    pos_weight = number of negative timesteps / number of positive timesteps

    This is computed from actual data rather than hardcoded so it
    adapts automatically if your anomaly rate changes.

    Args:
        windows : list of window dicts from phase2_preprocessing.build_windows()
        device  : torch device to place the tensor on

    Returns:
        pos_weight : scalar tensor
    """
    all_labels = np.concatenate([w['y'] for w in windows])
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos

    if n_pos == 0:
        raise ValueError("No positive (anomaly) samples found in training windows.")

    weight = n_neg / n_pos
    print(f"Class balance  → normal: {n_neg:,.0f}  anomalous: {n_pos:,.0f}")
    print(f"pos_weight     → {weight:.3f}  "
          f"(missing an anomaly costs {weight:.1f}x a false alarm)")

    return torch.tensor([weight], dtype=torch.float32, device=device)


class WeightedBCELoss(nn.Module):
    """
    Thin wrapper around BCEWithLogitsLoss that bundles the pos_weight.

    Expects raw logits (before sigmoid) from the model's
    forward(return_logits=True) call.

    Args:
        pos_weight : scalar tensor from compute_pos_weight()
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,   # (batch, T) — raw model output
        labels: torch.Tensor,   # (batch, T) — 0.0 or 1.0
    ) -> torch.Tensor:
        return self.loss_fn(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cpu')

    # Simulate windows with ~30% anomaly rate
    rng = np.random.default_rng(42)
    fake_windows = [
        {'y': rng.choice([0.0, 1.0], size=15, p=[0.70, 0.30]).astype(np.float32)}
        for _ in range(500)
    ]

    pos_weight = compute_pos_weight(fake_windows, device)
    loss_fn    = WeightedBCELoss(pos_weight)

    # Simulate a batch
    BATCH, T = 32, 15
    logits = torch.randn(BATCH, T)
    labels = torch.zeros(BATCH, T)
    labels[:, 8:] = 1.0    # last 7 timesteps are anomalous

    loss = loss_fn(logits, labels)

    print(f"\nSample loss value : {loss.item():.4f}")
    print(f"Loss is scalar    : {loss.shape == torch.Size([])}")
    assert loss.shape == torch.Size([]), "Loss must be a scalar"
    assert loss.item() > 0,             "Loss must be positive"

    # Verify weighting is working:
    # A model that always predicts "anomaly" should get lower
    # loss than one that always predicts "normal" when anomaly
    # rate is non-trivial and pos_weight > 1.
    always_anomaly = loss_fn(torch.ones(BATCH, T)  * 5.0, labels)
    always_normal  = loss_fn(torch.ones(BATCH, T) * -5.0, labels)

    print(f"\nLoss if always predicts anomaly : {always_anomaly.item():.4f}")
    print(f"Loss if always predicts normal  : {always_normal.item():.4f}")
    print("pos_weight is penalising missed anomalies correctly:",
          always_normal.item() > always_anomaly.item())
    print('\nLoss function ready.')

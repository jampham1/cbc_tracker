"""
Phase 4 — Part 2: Training Loop
=================================
Implements the train/validation cycle with:
  - AdamW optimiser with weight decay
  - ReduceLROnPlateau learning rate scheduler
  - Early stopping on validation loss
  - Best model checkpointing
  - Per-epoch metrics: loss, AUROC, sensitivity, specificity
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    all_probs:  np.ndarray,   # (N,) predicted probabilities
    all_labels: np.ndarray,   # (N,) true binary labels
    threshold:  float = 0.5,
) -> dict:
    """
    Compute AUROC, sensitivity, and specificity.

    Sensitivity (recall) = TP / (TP + FN)
        → of all real anomalies, what fraction did we catch?
        → the critical clinical metric: missed anomalies are dangerous

    Specificity = TN / (TN + FP)
        → of all normal draws, what fraction did we correctly clear?
        → too many false alarms erodes clinician trust

    AUROC is threshold-independent — it measures overall discrimination.
    A random model scores 0.5; a perfect model scores 1.0.
    """
    # AUROC — requires both classes to be present
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float('nan')

    # Threshold-based metrics
    preds = (all_probs >= threshold).astype(float)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'auroc':       auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimiser,
    device,
) -> dict:
    """
    Run one full pass over the training DataLoader.
    Returns a dict of metrics for this epoch.
    """
    model.train()

    total_loss  = 0.0
    all_probs   = []
    all_labels  = []

    for x_seq, x_static, y in loader:
        x_seq    = x_seq.to(device)
        x_static = x_static.to(device)
        y        = y.to(device)

        optimiser.zero_grad()

        # Forward pass — return logits for numerically stable loss
        logits = model(x_seq, x_static, return_logits=True)
        loss   = loss_fn(logits, y)

        # Backward pass
        loss.backward()

        # Gradient clipping — prevents exploding gradients.
        # If the gradient norm exceeds max_norm, all gradients
        # are scaled down proportionally. This is especially
        # important in deep networks during early training.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()

        # Accumulate metrics
        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy().ravel())
        all_labels.append(y.cpu().numpy().ravel())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(loader)

    metrics = compute_metrics(all_probs, all_labels)
    metrics['loss'] = avg_loss
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of validation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device) -> dict:
    """
    Run one full pass over the validation DataLoader.
    No gradients computed — purely for monitoring.
    """
    model.eval()

    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for x_seq, x_static, y in loader:
        x_seq    = x_seq.to(device)
        x_static = x_static.to(device)
        y        = y.to(device)

        logits = model(x_seq, x_static, return_logits=True)
        loss   = loss_fn(logits, y)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy().ravel())
        all_labels.append(y.cpu().numpy().ravel())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(loader)

    metrics = compute_metrics(all_probs, all_labels)
    metrics['loss'] = avg_loss
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why early stopping?
        Without it, the model will eventually memorise the training
        set — it fits the noise in the data rather than the signal.
        This is called overfitting. The model gets better on training
        data but worse on unseen patients.

        Early stopping monitors the val loss and halts training when
        it hasn't improved for `patience` consecutive epochs, then
        restores the best weights seen so far.

    Args:
        patience  : epochs to wait for improvement before stopping
        min_delta : minimum change to count as an improvement
        ckpt_path : where to save the best model weights
    """

    def __init__(
        self,
        patience:  int   = 10,
        min_delta: float = 1e-4,
        ckpt_path: str   = 'outputs/checkpoints/best_model.pt',
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.ckpt_path  = Path(ckpt_path)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss    = float('inf')
        self.epochs_waited = 0
        self.stopped       = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call after each validation epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found — save checkpoint and reset counter
            self.best_loss     = val_loss
            self.epochs_waited = 0
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            self.epochs_waited += 1
            if self.epochs_waited >= self.patience:
                self.stopped = True

        return self.stopped

    def load_best(self, model: nn.Module, device: torch.device):
        """Restore the best weights after training completes."""
        model.load_state_dict(torch.load(self.ckpt_path, map_location=device))
        print(f"Restored best model from {self.ckpt_path} "
              f"(val_loss = {self.best_loss:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Full training run
# ─────────────────────────────────────────────────────────────────────────────
def run_training(
    model,
    train_loader,
    val_loader,
    train_windows,
    device,
    n_epochs:    int   = 50,
    lr:          float = 1e-3,
    weight_decay: float = 1e-4,
    patience:    int   = 10,
    ckpt_path:   str   = 'outputs/checkpoints/best_model.pt',
) -> dict:
    """
    Full training run. Returns history dict of per-epoch metrics.

    Optimiser — AdamW:
        Adam with decoupled weight decay. Weight decay adds a small
        penalty proportional to the weight magnitude, discouraging
        the model from memorising training data. 'Decoupled' means
        the decay is applied directly to the weights, not mixed into
        the gradient — which is more principled than standard Adam.

    Scheduler — ReduceLROnPlateau:
        Halves the learning rate when val loss plateaus for 5 epochs.
        Starts at lr=1e-3, can reduce down to lr=1e-6.
        This lets the model take big steps early and fine-tune later.
    """
    from phase4_part1_loss import compute_pos_weight, WeightedBCELoss

    # Loss
    pos_weight = compute_pos_weight(train_windows, device)
    loss_fn    = WeightedBCELoss(pos_weight)

    # Optimiser
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = 'min',     # reduce when val loss stops decreasing
        factor   = 0.5,       # halve the LR
        patience = 5,         # wait 5 epochs before reducing
        min_lr   = 1e-6,
    )

    # Early stopping
    stopper = EarlyStopping(patience=patience, ckpt_path=ckpt_path)

    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_specificity': [], 'val_specificity': [],
        'lr': [],
    }

    print(f"\nTraining on {device} for up to {n_epochs} epochs "
          f"(early stop patience={patience})")
    print(f"{'Epoch':<7} {'Train loss':<12} {'Val loss':<12} "
          f"{'Val AUROC':<11} {'Val sens':<10} {'Val spec':<10} {'LR'}")
    print('─' * 78)

    model = model.to(device)

    for epoch in range(1, n_epochs + 1):
        train_m = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        val_m   = validate_one_epoch(model, val_loader, loss_fn, device)

        current_lr = optimiser.param_groups[0]['lr']
        scheduler.step(val_m['loss'])

        # Log
        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['train_auroc'].append(train_m['auroc'])
        history['val_auroc'].append(val_m['auroc'])
        history['train_sensitivity'].append(train_m['sensitivity'])
        history['val_sensitivity'].append(val_m['sensitivity'])
        history['train_specificity'].append(train_m['specificity'])
        history['val_specificity'].append(val_m['specificity'])
        history['lr'].append(current_lr)

        print(
            f"{epoch:<7} "
            f"{train_m['loss']:<12.4f} "
            f"{val_m['loss']:<12.4f} "
            f"{val_m['auroc']:<11.4f} "
            f"{val_m['sensitivity']:<10.4f} "
            f"{val_m['specificity']:<10.4f} "
            f"{current_lr:.2e}"
        )

        # Early stopping check
        if stopper.step(val_m['loss'], model):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best weights
    stopper.load_best(model, device)
    print(f'\nTraining complete. Best val loss: {stopper.best_loss:.4f}')

    return history

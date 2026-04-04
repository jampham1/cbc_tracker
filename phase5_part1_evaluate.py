"""
Phase 5 — Part 1: Test Set Evaluation
=======================================
Loads the best saved model and evaluates it on the held-out test set.

Metrics computed:
  - AUROC           : overall discrimination ability
  - AUPRC           : area under precision-recall curve
                      more informative than AUROC on imbalanced data
  - Sensitivity     : fraction of real anomalies caught  (recall)
  - Specificity     : fraction of normals correctly cleared
  - PPV             : positive predictive value (precision)
                      of flagged draws, how many are truly anomalous?
  - F1              : harmonic mean of precision and recall
  - Confusion matrix: TP, TN, FP, FN counts

Results are saved to outputs/evaluation_results.json for the notebook.
"""

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pass
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test_set(
    model,
    test_loader,
    test_windows: list,
    device:       torch.device,
    threshold:    float = 0.5,
) -> dict:
    """
    Run inference over the full test set and compute all metrics.

    Returns a results dict containing:
      - scalar metrics (auroc, auprc, sensitivity, etc.)
      - arrays for plotting (fpr, tpr, precision, recall curves)
      - per-window anomaly scores for patient-level analysis
      - confusion matrix
    """
    model.eval()
    model = model.to(device)

    all_probs  = []
    all_labels = []

    for x_seq, x_static, y in test_loader:
        x_seq    = x_seq.to(device)
        x_static = x_static.to(device)

        probs = model(x_seq, x_static, return_logits=False)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

    all_probs  = np.concatenate(all_probs).ravel()    # (N_timesteps,)
    all_labels = np.concatenate(all_labels).ravel()   # (N_timesteps,)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    preds = (all_probs >= threshold).astype(float)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # recall
    specificity = tn / (tn + fp + 1e-8)
    ppv         = tp / (tp + fp + 1e-8)   # precision
    npv         = tn / (tn + fn + 1e-8)
    f1          = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)

    # ── Curve arrays (for plotting) ───────────────────────────────────────────
    fpr, tpr, roc_thresh  = roc_curve(all_labels, all_probs)
    prec, rec, pr_thresh  = precision_recall_curve(all_labels, all_probs)

    # ── Per-window max anomaly score (patient-level view) ─────────────────────
    # Reconstruct per-window scores from the flat array.
    # Each window has T=window_size timesteps.
    T = test_windows[0]['x_seq'].shape[0]
    n_windows = len(test_windows)
    probs_2d  = all_probs[:n_windows * T].reshape(n_windows, T)

    window_scores = []
    for i, w in enumerate(test_windows):
        window_scores.append({
            'patient_id':   int(w['meta']['patient_id']),
            'anomaly_type': w['meta']['anomaly_type'],
            'max_score':    float(probs_2d[i].max()),
            'mean_score':   float(probs_2d[i].mean()),
            'true_positive': int(w['y'].max()),   # 1 if any anomaly in window
        })

    results = {
        # Scalars
        'auroc':       float(auroc),
        'auprc':       float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv':         float(ppv),
        'npv':         float(npv),
        'f1':          float(f1),
        'accuracy':    float(accuracy),
        'threshold':   threshold,

        # Confusion matrix
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),

        # Curve arrays (convert to list for JSON serialisation)
        'fpr':        fpr.tolist(),
        'tpr':        tpr.tolist(),
        'precision':  prec.tolist(),
        'recall':     rec.tolist(),

        # Per-window
        'window_scores': window_scores,

        # Raw arrays for notebook plotting
        'all_probs':  all_probs.tolist(),
        'all_labels': all_labels.tolist(),
    }

    return results


def print_results(results: dict):
    """Print a formatted clinical summary."""
    print('\n' + '═' * 52)
    print('  TEST SET EVALUATION RESULTS')
    print('═' * 52)
    print(f"  AUROC            : {results['auroc']:.4f}")
    print(f"  AUPRC            : {results['auprc']:.4f}")
    print()
    print(f"  Sensitivity      : {results['sensitivity']:.4f}"
          f"  ← fraction of anomalies caught")
    print(f"  Specificity      : {results['specificity']:.4f}"
          f"  ← fraction of normals cleared")
    print(f"  PPV (precision)  : {results['ppv']:.4f}"
          f"  ← of flagged draws, % truly anomalous")
    print(f"  NPV              : {results['npv']:.4f}")
    print(f"  F1 score         : {results['f1']:.4f}")
    print(f"  Accuracy         : {results['accuracy']:.4f}")
    print()
    print(f"  Confusion matrix (threshold = {results['threshold']}):")
    print(f"    TP={results['tp']:,}  FP={results['fp']:,}")
    print(f"    FN={results['fn']:,}  TN={results['tn']:,}")
    print('═' * 52)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    ckpt_path:  str = 'outputs/checkpoints/best_model.pt',
    output_dir: str = 'outputs',
):
    from phase2_preprocessing import run_preprocessing
    from phase3_part4_full_model import CBCAnomalyTCN

    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()         else
        torch.device('cpu')
    )

    # Preprocess
    data = run_preprocessing(csv_path='data/cbc_synthetic.csv',
                              output_dir=output_dir)

    # Load model
    model = CBCAnomalyTCN(
        n_seq_features    = data['n_seq_features'],
        n_static_features = data['n_static_features'],
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    print(f'Loaded model weights from {ckpt_path}')

    # Evaluate
    results = evaluate_test_set(
        model        = model,
        test_loader  = data['test_loader'],
        test_windows = data['test_windows'],
        device       = device,
    )

    print_results(results)

    # Save for notebook
    save_path = Path(output_dir) / 'evaluation_results.json'
    # Remove large arrays before saving to keep file manageable
    save_results = {k: v for k, v in results.items()
                    if k not in ('all_probs', 'all_labels')}
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'\nResults saved → {save_path}')

    return results


if __name__ == '__main__':
    run_evaluation()

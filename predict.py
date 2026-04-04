"""
Phase 5 — Part 2: predict.py
==============================
Standalone inference script. Given a CSV of CBC draws for a new patient,
returns a per-draw anomaly probability using the trained model.

This is the file you would hand to a clinician-facing system.
It requires NO knowledge of the training code — just the saved
model weights and preprocessor.

Usage:
    python predict.py --patient_csv path/to/patient_draws.csv

Expected CSV columns:
    patient_id, timestamp_day, time_delta,
    WBC, ANC, RBC, HGB, PLT, LYM, MONO,
    age, sex, cancer_type, chemo_protocol

Output:
    Printed table of draw-level anomaly probabilities.
    Saves to outputs/predictions/<patient_id>_predictions.csv
"""

import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from phase2_preprocessing import (
    CBCPreprocessor, build_windows,
    CBC_FEATURES, WINDOW_SIZE, STRIDE
)
from phase3_part4_full_model import CBCAnomalyTCN


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_patient(
    patient_df:   pd.DataFrame,
    model:        torch.nn.Module,
    preprocessor: CBCPreprocessor,
    device:       torch.device,
    window_size:  int = WINDOW_SIZE,
    stride:       int = 1,           # stride=1 for inference: score every draw
) -> pd.DataFrame:
    """
    Run anomaly detection on a single patient's CBC timeline.

    Key difference from training:
        stride=1 during inference — we produce a score at every possible
        timestep rather than every 3rd. This gives a continuous risk
        timeline the clinician can read like a chart.

    Returns a DataFrame with the original draws plus:
        anomaly_score  : model's probability this draw is anomalous
        risk_level     : 'low' / 'medium' / 'high' based on score
    """
    model.eval()
    patient_id = patient_df['patient_id'].iloc[0]

    # Preprocess using training-fitted statistics
    proc_df = preprocessor.transform(patient_df)

    # Build windows with stride=1
    windows = build_windows(proc_df, [patient_id], window_size, stride=1)

    if len(windows) == 0:
        print(f"  Warning: patient has fewer than {window_size} draws. "
              f"Cannot score.")
        return pd.DataFrame()

    # Run inference on each window, collect the score at the LAST timestep.
    # Why last timestep only?
    # Each window ends at a different draw. Taking the last position gives
    # us one score per draw, each conditioned on all preceding context.
    # This is the most clinically natural view — "given everything up to
    # draw t, how anomalous is draw t?"
    scores = {}   # draw_index → anomaly_score

    for w in windows:
        x_seq    = torch.from_numpy(w['x_seq']).unsqueeze(0).to(device)
        x_static = torch.from_numpy(w['x_static']).unsqueeze(0).to(device)

        prob = model(x_seq, x_static)   # (1, T)
        last_score = prob[0, -1].item()  # scalar — score at the last draw

        draw_idx = w['meta']['window_start'] + window_size - 1
        scores[draw_idx] = last_score

    # Build output DataFrame aligned to original draw indices
    result_df = patient_df.copy().reset_index(drop=True)
    result_df['anomaly_score'] = result_df.index.map(
        lambda i: scores.get(i, np.nan)
    )

    # Risk level thresholds
    def risk_label(score):
        if np.isnan(score): return 'insufficient data'
        if score >= 0.7:    return 'high'
        if score >= 0.4:    return 'medium'
        return 'low'

    result_df['risk_level'] = result_df['anomaly_score'].apply(risk_label)

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Load model + preprocessor from disk
# ─────────────────────────────────────────────────────────────────────────────
def load_inference_artifacts(
    ckpt_path:        str = 'outputs/checkpoints/best_model.pt',
    preprocessor_path: str = 'outputs/preprocessor.pkl',
    n_seq_features:   int = 8,
    n_static_features: int = 4,
) -> tuple:
    """
    Load saved model weights and fitted preprocessor.
    These two files are all you need to run inference on new data.
    """
    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()         else
        torch.device('cpu')
    )

    # Load preprocessor (carries training-set mean/std for normalisation)
    preprocessor = CBCPreprocessor.load(preprocessor_path)
    print(f'Preprocessor loaded from {preprocessor_path}')

    # Instantiate architecture and load weights
    model = CBCAnomalyTCN(
        n_seq_features    = n_seq_features,
        n_static_features = n_static_features,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f'Model weights loaded from {ckpt_path}')
    print(f'Running on: {device}')

    return model, preprocessor, device


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Run CBC anomaly detection on a patient CSV.'
    )
    parser.add_argument(
        '--patient_csv', type=str, required=True,
        help='Path to CSV file containing patient CBC draws'
    )
    parser.add_argument(
        '--ckpt', type=str,
        default='outputs/checkpoints/best_model.pt'
    )
    parser.add_argument(
        '--preprocessor', type=str,
        default='outputs/preprocessor.pkl'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='outputs/predictions'
    )
    args = parser.parse_args()

    # Load
    model, preprocessor, device = load_inference_artifacts(
        ckpt_path         = args.ckpt,
        preprocessor_path = args.preprocessor,
    )

    # Read patient data
    patient_df = pd.read_csv(args.patient_csv)
    patient_id = patient_df['patient_id'].iloc[0]
    print(f'\nScoring patient {patient_id} '
          f'({len(patient_df)} CBC draws)...')

    # Run prediction
    result_df = predict_patient(patient_df, model, preprocessor, device)

    if result_df.empty:
        return

    # Print summary
    display_cols = ['timestamp_day', 'WBC', 'ANC', 'PLT',
                    'anomaly_score', 'risk_level']
    print(f'\n{result_df[display_cols].to_string(index=False)}\n')

    high_risk = result_df[result_df['risk_level'] == 'high']
    if len(high_risk) > 0:
        print(f'  {len(high_risk)} HIGH RISK draws detected at days: '
              f'{high_risk["timestamp_day"].tolist()}')
    else:
        print('  No high-risk draws detected.')

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f'{args.output_dir}/patient_{patient_id}_predictions.csv'
    result_df.to_csv(save_path, index=False)
    print(f'\nPredictions saved → {save_path}')


if __name__ == '__main__':
    main()

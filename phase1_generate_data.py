"""
Phase 1: Synthetic CBC Data Generator
======================================
Simulates longitudinal CBC lab draws for cancer patients undergoing chemotherapy.

Each patient has:
  - Irregular time intervals between draws (realistic clinic scheduling)
  - Chemotherapy-induced nadir patterns (expected suppression of counts)
  - Injected anomalies: sudden drops, sustained elevation, erratic oscillation
  - Static covariates: age, sex, cancer type, chemo protocol

CBC Features generated:
  WBC   - White blood cell count (x10^9/L)
  ANC   - Absolute neutrophil count (x10^9/L)  [most critical in chemo]
  RBC   - Red blood cell count (x10^12/L)
  HGB   - Hemoglobin (g/dL)
  PLT   - Platelet count (x10^9/L)
  LYM   - Lymphocyte count (x10^9/L)
  MONO  - Monocyte count (x10^9/L)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Clinical reference ranges (approximate adult normals) ──────────────────────
NORMAL_RANGES = {
    "WBC":  (4.5,  11.0),
    "ANC":  (1.8,   7.7),
    "RBC":  (4.2,   5.4),
    "HGB":  (12.0, 17.5),
    "PLT":  (150,  400),
    "LYM":  (1.0,   4.8),
    "MONO": (0.2,   0.95),
}

CANCER_TYPES  = ["AML", "CLL", "NHL", "MM", "Breast-CA"]
CHEMO_PROTOS  = ["R-CHOP", "FOLFOX", "BEP", "VAD", "AC-T"]
ANOMALY_TYPES = ["sudden_drop", "sustained_elevation", "erratic", "none"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate a single feature trajectory
# ─────────────────────────────────────────────────────────────────────────────
def _baseline(feature: str) -> float:
    """Sample a healthy baseline value from the normal range."""
    lo, hi = NORMAL_RANGES[feature]
    return rng.uniform(lo * 0.9, hi * 1.1)


def _chemo_nadir_multiplier(t: np.ndarray, cycle_len: int = 21) -> np.ndarray:
    """
    Simulate a chemotherapy nadir pattern.
    Counts drop around day 10-14 of each cycle and recover by day 21.
    Returns a multiplier in [0.2, 1.0].
    """
    phase   = (t % cycle_len) / cycle_len          # 0→1 within each cycle
    # Nadir trough centred at 0.5 (day ~10-11), shaped as inverted Gaussian
    nadir   = 1.0 - 0.65 * np.exp(-0.5 * ((phase - 0.5) / 0.15) ** 2)
    noise   = rng.normal(0, 0.03, size=t.shape)
    return np.clip(nadir + noise, 0.15, 1.05)


def _generate_trajectory(
    feature: str,
    timestamps: np.ndarray,
    anomaly_type: str,
    anomaly_start_idx: int,
) -> np.ndarray:
    """
    Build one CBC feature trajectory across `timestamps` (days from diagnosis).
    """
    n        = len(timestamps)
    baseline = _baseline(feature)
    lo, hi   = NORMAL_RANGES[feature]

    # Chemo-driven nadir on top of baseline
    nadir_mult = _chemo_nadir_multiplier(timestamps)
    # Slow trend: mild decline over time (treatment burden)
    trend      = 1.0 - 0.0015 * timestamps
    # Measurement noise
    noise      = rng.normal(0, (hi - lo) * 0.04, size=n)

    values = baseline * nadir_mult * trend + noise

    # ── Inject anomaly ────────────────────────────────────────────────────────
    if anomaly_type != "none" and anomaly_start_idx < n:
        idx = anomaly_start_idx

        if anomaly_type == "sudden_drop":
            # Sharp drop to 20–40% of normal over 2–4 draws, then partial recovery
            drop_len = min(rng.integers(2, 5), n - idx)
            for k in range(drop_len):
                values[idx + k] = baseline * rng.uniform(0.15, 0.35)
            # slow partial recovery
            for k in range(drop_len, min(drop_len + 4, n - idx)):
                values[idx + k] = baseline * rng.uniform(0.4, 0.7)

        elif anomaly_type == "sustained_elevation":
            # Counts climb and stay above upper limit — could indicate infection/relapse
            elev_len = min(rng.integers(4, 10), n - idx)
            for k in range(elev_len):
                values[idx + k] = hi * rng.uniform(1.3, 2.2)

        elif anomaly_type == "erratic":
            # High-variance oscillation — treatment failure / instability
            erratic_len = min(rng.integers(5, 12), n - idx)
            for k in range(erratic_len):
                values[idx + k] = rng.uniform(lo * 0.2, hi * 2.5)

    return np.clip(values, 0.0, None)   # physiologically non-negative


# ─────────────────────────────────────────────────────────────────────────────
# Core: generate one patient
# ─────────────────────────────────────────────────────────────────────────────
def generate_patient(patient_id: int) -> pd.DataFrame:
    """
    Returns a DataFrame of longitudinal CBC draws for one patient.

    Columns:
        patient_id, timestamp_day, time_delta,
        WBC, ANC, RBC, HGB, PLT, LYM, MONO,
        anomaly_type, is_anomaly,
        age, sex, cancer_type, chemo_protocol
    """
    # Static covariates
    age           = int(rng.integers(28, 82))
    sex           = rng.choice(["M", "F"])
    cancer_type   = rng.choice(CANCER_TYPES)
    chemo_proto   = rng.choice(CHEMO_PROTOS)

    # Number of CBC draws: 20–60 (irregular follow-up length)
    n_draws = int(rng.integers(20, 61))

    # Irregular timestamps: clinical draws cluster around chemo visits
    # Base: every ~7 days with jitter ±3 days
    inter_draw_gaps = rng.integers(3, 14, size=n_draws - 1)
    timestamps      = np.concatenate([[0], np.cumsum(inter_draw_gaps)]).astype(float)

    # Anomaly setup
    anomaly_type = rng.choice(ANOMALY_TYPES, p=[0.20, 0.15, 0.15, 0.50])
    # Anomaly starts after at least 30% of the timeline
    anomaly_start_idx = int(rng.integers(max(1, n_draws // 3), n_draws - 5)) \
                        if anomaly_type != "none" else n_draws

    # Generate each feature
    records = []
    feature_arrays = {
        feat: _generate_trajectory(feat, timestamps, anomaly_type, anomaly_start_idx)
        for feat in NORMAL_RANGES
    }

    for i in range(n_draws):
        is_anomaly = (anomaly_type != "none") and (i >= anomaly_start_idx)
        record = {
            "patient_id":     patient_id,
            "timestamp_day":  timestamps[i],
            "time_delta":     timestamps[i] - timestamps[i - 1] if i > 0 else 0.0,
            **{feat: feature_arrays[feat][i] for feat in NORMAL_RANGES},
            "anomaly_type":   anomaly_type,
            "is_anomaly":     int(is_anomaly),
            # Static features (repeated per row for convenience)
            "age":            age,
            "sex":            sex,
            "cancer_type":    cancer_type,
            "chemo_protocol": chemo_proto,
        }
        records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main: generate full cohort
# ─────────────────────────────────────────────────────────────────────────────
def generate_cohort(n_patients: int = 500, save_path: str = "data/cbc_synthetic.csv"):
    """
    Generate a cohort of `n_patients` patients and save to CSV.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    all_dfs = [generate_patient(pid) for pid in range(n_patients)]
    cohort  = pd.concat(all_dfs, ignore_index=True)

    # Round CBC values to realistic precision
    for feat in ["WBC", "ANC", "LYM", "MONO"]:
        cohort[feat] = cohort[feat].round(2)
    for feat in ["RBC"]:
        cohort[feat] = cohort[feat].round(2)
    for feat in ["HGB"]:
        cohort[feat] = cohort[feat].round(1)
    cohort["PLT"] = cohort["PLT"].round(0).astype(int)

    cohort.to_csv(save_path, index=False)
    return cohort


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def describe_cohort(df: pd.DataFrame):
    print("=" * 60)
    print(f"  Cohort: {df['patient_id'].nunique()} patients, "
          f"{len(df):,} total CBC draws")
    print("=" * 60)

    print("\n── Anomaly distribution ──────────────────────────────────")
    anom_counts = (
        df.drop_duplicates("patient_id")["anomaly_type"]
        .value_counts()
    )
    for atype, cnt in anom_counts.items():
        pct = cnt / anom_counts.sum() * 100
        print(f"  {atype:<25} {cnt:>4} patients  ({pct:.1f}%)")

    print("\n── CBC feature stats (all draws) ─────────────────────────")
    feat_cols = list(NORMAL_RANGES.keys())
    stats = df[feat_cols].describe().loc[["mean", "std", "min", "max"]]
    print(stats.round(2).to_string())

    print("\n── Draws per patient ─────────────────────────────────────")
    draws_per_pt = df.groupby("patient_id").size()
    print(f"  min={draws_per_pt.min()}  "
          f"median={draws_per_pt.median():.0f}  "
          f"max={draws_per_pt.max()}")

    print("\n── Anomalous draw fraction ───────────────────────────────")
    frac = df["is_anomaly"].mean() * 100
    print(f"  {frac:.1f}% of all draws are labelled anomalous")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic CBC cohort...")
    cohort = generate_cohort(n_patients=500, save_path="data/cbc_synthetic.csv")
    describe_cohort(cohort)
    print("\nSaved → data/cbc_synthetic.csv")

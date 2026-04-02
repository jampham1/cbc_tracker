"""
Phase 2: Preprocessing Pipeline
=================================
Transforms raw longitudinal CBC data into model-ready PyTorch tensors.

Steps performed:
  1. Load & validate raw CSV from Phase 1
  2. Encode categorical static features (sex, cancer type, chemo protocol)
  3. Normalise CBC features per-feature using training-set statistics only
     (fitted on train split, applied to val/test — prevents data leakage)
  4. Encode time_delta as an explicit input feature (log-scaled)
  5. Construct sliding windows of fixed length T over each patient's timeline
  6. Split patients (not rows) into train / val / test sets
  7. Wrap everything in a PyTorch Dataset + DataLoader

Output tensors per window:
  x_seq   : (T, n_cbc_features + 1)   ← CBC values + log time_delta
  x_static: (n_static_features,)       ← age, sex encoding, cancer type, chemo
  y       : (T,)                        ← is_anomaly label per timestep
  meta    : dict with patient_id, window start index (for evaluation)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")


# ── Constants ──────────────────────────────────────────────────────────────────
CBC_FEATURES    = ["WBC", "ANC", "RBC", "HGB", "PLT", "LYM", "MONO"]
STATIC_CAT_COLS = ["sex", "cancer_type", "chemo_protocol"]
STATIC_NUM_COLS = ["age"]
WINDOW_SIZE     = 15    # T: number of consecutive draws per training window
STRIDE          = 3     # step between windows (overlap is fine and increases data)
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
# TEST_FRAC     = 0.15  (remainder)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2 — Load and encode categoricals
# ─────────────────────────────────────────────────────────────────────────────
class CBCPreprocessor:
    """
    Fits normalisation statistics and label encoders on the training split.
    Must call .fit(train_df) before .transform(df).

    Keeps all state so it can be serialised and reloaded for inference.
    """

    def __init__(self):
        self.cbc_means_   = None   # dict: feature → mean  (from train)
        self.cbc_stds_    = None   # dict: feature → std
        self.label_encs_  = {}     # dict: col → fitted LabelEncoder
        self.age_mean_    = None
        self.age_std_     = None
        self.fitted_      = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "CBCPreprocessor":
        """Learn statistics from training data only."""

        # CBC z-score parameters
        self.cbc_means_ = {f: train_df[f].mean() for f in CBC_FEATURES}
        self.cbc_stds_  = {f: max(float(train_df[f].std()), 1e-6) for f in CBC_FEATURES}

        # Age normalisation
        self.age_mean_ = train_df["age"].mean()
        self.age_std_  = max(float(train_df["age"].std()), 1e-6)

        # Categorical label encoders
        for col in STATIC_CAT_COLS:
            le = LabelEncoder()
            le.fit(train_df[col].astype(str))
            self.label_encs_[col] = le

        self.fitted_ = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to any split."""
        assert self.fitted_, "Call .fit() on training data before .transform()"
        df = df.copy()

        # Normalise CBC features
        for f in CBC_FEATURES:
            df[f] = (df[f] - self.cbc_means_[f]) / self.cbc_stds_[f]

        # Log-scale time_delta: log(1 + delta_days)
        # This compresses large gaps and makes 0-day (first draw) sensible
        df["log_time_delta"] = np.log1p(df["time_delta"])

        # Normalise age
        df["age_norm"] = (df["age"] - self.age_mean_) / self.age_std_

        # Encode categoricals to integers
        for col in STATIC_CAT_COLS:
            le = self.label_encs_[col]
            # Handle unseen categories gracefully (map to 0)
            df[col + "_enc"] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else 0
            )

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "CBCPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Patient-level train / val / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_patients(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    seed:       int   = 42,
) -> tuple[list, list, list]:
    """
    Split at the PATIENT level (not the row level).
    This is critical — splitting rows would let the model see future draws from
    the same patient during training, which is data leakage.

    Returns three lists of patient_ids: train, val, test.
    """
    all_pids = df["patient_id"].unique()
    rng      = np.random.default_rng(seed)
    rng.shuffle(all_pids)

    n        = len(all_pids)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)

    train_ids = all_pids[:n_train].tolist()
    val_ids   = all_pids[n_train : n_train + n_val].tolist()
    test_ids  = all_pids[n_train + n_val :].tolist()

    print(f"Split → train: {len(train_ids)} pts | "
          f"val: {len(val_ids)} pts | "
          f"test: {len(test_ids)} pts")
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Sliding window construction
# ─────────────────────────────────────────────────────────────────────────────
def build_windows(
    df: pd.DataFrame,
    patient_ids: list,
    window_size: int = WINDOW_SIZE,
    stride:      int = STRIDE,
) -> list[dict]:
    """
    For each patient, slide a window of length `window_size` across their
    draw timeline with step `stride`. Returns a list of window dicts.

    Each window dict contains:
        x_seq   : np.ndarray (window_size, n_seq_features)
        x_static: np.ndarray (n_static_features,)
        y       : np.ndarray (window_size,)   ← per-timestep anomaly label
        meta    : dict
    """
    seq_features    = CBC_FEATURES + ["log_time_delta"]
    static_features = ["age_norm"] + [c + "_enc" for c in STATIC_CAT_COLS]

    windows = []
    skipped = 0

    for pid in patient_ids:
        pt_df = df[df["patient_id"] == pid].sort_values("timestamp_day").reset_index(drop=True)

        if len(pt_df) < window_size:
            skipped += 1
            continue

        # Static features: same for every window from this patient
        x_static = pt_df[static_features].iloc[0].values.astype(np.float32)

        # Slide window
        for start in range(0, len(pt_df) - window_size + 1, stride):
            end    = start + window_size
            window = pt_df.iloc[start:end]

            x_seq = window[seq_features].values.astype(np.float32)  # (T, F)
            y     = window["is_anomaly"].values.astype(np.float32)   # (T,)

            windows.append({
                "x_seq":    x_seq,
                "x_static": x_static,
                "y":        y,
                "meta": {
                    "patient_id":   pid,
                    "window_start": start,
                    "anomaly_type": pt_df["anomaly_type"].iloc[0],
                },
            })

    print(f"Built {len(windows):,} windows "
          f"(skipped {skipped} patients with < {window_size} draws)")
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — PyTorch Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class CBCWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of window dicts from build_windows().

    __getitem__ returns:
        x_seq   : FloatTensor (T, n_seq_features)
        x_static: FloatTensor (n_static_features,)
        y       : FloatTensor (T,)
    """

    def __init__(self, windows: list[dict]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return (
            torch.from_numpy(w["x_seq"]),       # (T, F)
            torch.from_numpy(w["x_static"]),    # (S,)
            torch.from_numpy(w["y"]),            # (T,)
        )


def make_loaders(
    train_windows: list,
    val_windows:   list,
    test_windows:  list,
    batch_size:    int = 32,
    num_workers:   int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders."""

    train_loader = DataLoader(
        CBCWindowDataset(train_windows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CBCWindowDataset(val_windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        CBCWindowDataset(test_windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Master function: run full pipeline from CSV path
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing(
    csv_path:    str = "data/cbc_synthetic.csv",
    output_dir:  str = "outputs/",
    window_size: int = WINDOW_SIZE,
    stride:      int = STRIDE,
    batch_size:  int = 32,
) -> dict:
    """
    End-to-end preprocessing. Returns a dict with loaders, preprocessor,
    and window lists for downstream use.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  {df['patient_id'].nunique()} patients, {len(df):,} draws loaded")

    # ── Split patients ────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = split_patients(df)

    train_df = df[df["patient_id"].isin(train_ids)]
    val_df   = df[df["patient_id"].isin(val_ids)]
    test_df  = df[df["patient_id"].isin(test_ids)]

    # ── Fit preprocessor on train only ───────────────────────────────────────
    print("\nFitting preprocessor on training split...")
    preprocessor = CBCPreprocessor()
    train_df_t   = preprocessor.fit_transform(train_df)
    val_df_t     = preprocessor.transform(val_df)
    test_df_t    = preprocessor.transform(test_df)

    # Save preprocessor for later inference use
    preprocessor.save(f"{output_dir}/preprocessor.pkl")

    # ── Build windows ─────────────────────────────────────────────────────────
    print("\nBuilding sliding windows...")
    train_windows = build_windows(train_df_t, train_ids, window_size, stride)
    val_windows   = build_windows(val_df_t,   val_ids,   window_size, stride)
    test_windows  = build_windows(test_df_t,  test_ids,  window_size, stride)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        train_windows, val_windows, test_windows, batch_size
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Preprocessing complete ─────────────────────────────────")
    print(f"  Window size : {window_size} draws")
    print(f"  Stride      : {stride} draws")
    print(f"  Seq features: {len(CBC_FEATURES) + 1}  "
          f"({', '.join(CBC_FEATURES)}, log_time_delta)")
    print(f"  Static feats: {1 + len(STATIC_CAT_COLS)}  "
          f"(age_norm, sex_enc, cancer_type_enc, chemo_protocol_enc)")
    print(f"  Train windows: {len(train_windows):,}")
    print(f"  Val   windows: {len(val_windows):,}")
    print(f"  Test  windows: {len(test_windows):,}")

    # Quick batch shape check
    x_seq, x_static, y = next(iter(train_loader))
    print(f"\n  Sample batch shapes:")
    print(f"    x_seq    : {tuple(x_seq.shape)}   ← (batch, T, features)")
    print(f"    x_static : {tuple(x_static.shape)}  ← (batch, static_features)")
    print(f"    y        : {tuple(y.shape)}   ← (batch, T)")
    print("=" * 55)

    return {
        "train_loader":  train_loader,
        "val_loader":    val_loader,
        "test_loader":   test_loader,
        "train_windows": train_windows,
        "val_windows":   val_windows,
        "test_windows":  test_windows,
        "preprocessor":  preprocessor,
        "n_seq_features": len(CBC_FEATURES) + 1,
        "n_static_features": 1 + len(STATIC_CAT_COLS),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_preprocessing(
        csv_path   = "data/cbc_synthetic.csv",
        output_dir = "outputs",
        batch_size = 32,
    )

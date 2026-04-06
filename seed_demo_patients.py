"""
seed_demo_patients.py  — fixed version
"""
import json, numpy as np
from pathlib import Path

PATIENTS_FILE = "data/patients_db.json"
rng = np.random.default_rng(42)

NORMAL_RANGES = {
    "WBC":  (4.5,  11.0), "ANC":  (1.8,   7.7), "RBC":  (4.2,  5.4),
    "HGB":  (12.0, 17.5), "PLT":  (150,   400),  "LYM":  (1.0,  4.8),
    "MONO": (0.2,   0.95),
}
CBC_FEATURES = list(NORMAL_RANGES.keys())

def _baseline(feat):
    lo, hi = NORMAL_RANGES[feat]
    mid = (lo + hi) / 2
    return float(rng.uniform(mid * 0.85, mid * 1.15))

def _nadir_mult(t, cycle_len=21, depth=0.55):
    phase = (t % cycle_len) / cycle_len
    return float(np.clip(
        1.0 - depth * np.exp(-0.5*((phase-0.5)/0.14)**2) + rng.normal(0,0.025),
        0.12, 1.06
    ))

def _make_timestamps(n_draws, min_gap=5, max_gap=12):
    """
    Realistic irregular spacing — gaps of 5-12 days, total ~120-200 days.
    BUG FIX: was using compounding range() causing 500+ day timelines.
    """
    gaps = [int(x) for x in rng.integers(min_gap, max_gap+1, size=n_draws-1)]
    ts   = [0]
    for g in gaps:
        ts.append(ts[-1] + g)
    return ts

def _normal_traj(feat, base, timestamps, nadir_depth=0.55):
    lo, hi = NORMAL_RANGES[feat]
    return [
        max(0.01, base * _nadir_mult(t, depth=nadir_depth)
            + float(rng.normal(0, (hi-lo)*0.03)))
        for t in timestamps
    ]

def _sudden_drop_traj(feat, base, timestamps, anom_start, drop_depth=0.15):
    """
    BUG FIX: drop is now IMMEDIATE — falls to floor on first anomaly draw,
    not a gradual linear descent. This produces a clear discontinuity that
    the TCN window will capture even with only a few post-drop draws.
    """
    lo, hi = NORMAL_RANGES[feat]
    floor  = base * drop_depth
    vals   = []
    for i, t in enumerate(timestamps):
        noise = float(rng.normal(0, (hi-lo)*0.03))
        rel   = i - anom_start
        if i < anom_start:
            val = base * _nadir_mult(t) + noise
        elif rel == 0:
            val = floor * rng.uniform(0.90, 1.10)      # sharp drop
        elif rel < 6:
            val = floor * rng.uniform(0.85, 1.25) + noise  # stays low
        elif rel < 12:
            # slow partial recovery — reaches ~55% of baseline
            progress = (rel - 6) / 6
            val = floor + (base*0.55 - floor)*progress + noise
        else:
            val = base * rng.uniform(0.45, 0.65) + noise
        vals.append(max(0.01, val))
    return vals

def _elevation_traj(feat, base, timestamps, anom_start, elev_factor=1.7):
    lo, hi = NORMAL_RANGES[feat]
    target = hi * elev_factor
    vals   = []
    for i, t in enumerate(timestamps):
        noise = float(rng.normal(0, (hi-lo)*0.04))
        rel   = i - anom_start
        if i < anom_start:
            val = base * _nadir_mult(t) + noise
        elif rel == 0:
            val = target * rng.uniform(0.95, 1.05)     # immediate jump
        else:
            val = target * rng.uniform(0.88, 1.16) + noise
        vals.append(max(0.01, val))
    return vals

def _erratic_traj(feat, base, timestamps, anom_start):
    lo, hi = NORMAL_RANGES[feat]
    vals  = []
    phase = 0
    flip  = int(rng.integers(2, 4))
    for i, t in enumerate(timestamps):
        noise = float(rng.normal(0, (hi-lo)*0.04))
        if i < anom_start:
            val = base * _nadir_mult(t) + noise
        else:
            flip -= 1
            if flip <= 0:
                phase = 1 - phase
                flip  = int(rng.integers(2, 4))
            if phase == 0:
                val = base * rng.uniform(0.10, 0.30) + noise  # suppressed
            else:
                val = hi   * rng.uniform(1.30, 2.50) + noise  # elevated
            if rng.random() < 0.12:
                val *= rng.uniform(0.3, 3.0)
        vals.append(max(0.01, val))
    return vals

def build_patient(pid, name, age, sex, cancer_type, chemo_protocol,
                  anomaly_type, n_draws=30, anom_start=15,
                  nadir_depth=0.55, drop_depth=0.15, elev_factor=1.7):
    baselines  = {f: _baseline(f) for f in CBC_FEATURES}
    timestamps = _make_timestamps(n_draws)

    trajs = {}
    for feat in CBC_FEATURES:
        b = baselines[feat]
        if   anomaly_type == "sudden_drop":
            trajs[feat] = _sudden_drop_traj(feat, b, timestamps, anom_start, drop_depth)
        elif anomaly_type == "sustained_elevation":
            trajs[feat] = _elevation_traj(feat, b, timestamps, anom_start, elev_factor)
        elif anomaly_type == "erratic":
            trajs[feat] = _erratic_traj(feat, b, timestamps, anom_start)
        else:
            trajs[feat] = _normal_traj(feat, b, timestamps, nadir_depth)

    for feat in ["WBC","ANC","LYM","MONO","RBC"]:
        trajs[feat] = [round(v,2) for v in trajs[feat]]
    trajs["HGB"] = [round(v,1) for v in trajs["HGB"]]
    trajs["PLT"] = [int(round(v,0)) for v in trajs["PLT"]]

    draws = []
    for i, t in enumerate(timestamps):
        d = {"patient_id": pid, "timestamp_day": int(t),
             "time_delta": int(t-timestamps[i-1]) if i>0 else 0,
             "age": age, "sex": sex, "cancer_type": cancer_type,
             "chemo_protocol": chemo_protocol}
        for feat in CBC_FEATURES:
            d[feat] = trajs[feat][i]
        draws.append(d)

    return {"name": name, "age": age, "sex": sex,
            "cancer_type": cancer_type, "chemo_protocol": chemo_protocol,
            "created_at": "2024-01-15T09:00:00", "draws": draws,
            "_anomaly_type": anomaly_type}

DEMO_PATIENTS = [
    # sudden_drop x4
    dict(pid="PT0001", name="Margaret Holloway", age=62, sex="F",
         cancer_type="AML",       chemo_protocol="VAD",
         anomaly_type="sudden_drop", n_draws=30, anom_start=15, drop_depth=0.12),
    dict(pid="PT0002", name="David Okafor",      age=55, sex="M",
         cancer_type="NHL",       chemo_protocol="R-CHOP",
         anomaly_type="sudden_drop", n_draws=28, anom_start=16, drop_depth=0.18),
    dict(pid="PT0003", name="Yuki Tanaka",        age=41, sex="F",
         cancer_type="Breast-CA", chemo_protocol="AC-T",
         anomaly_type="sudden_drop", n_draws=32, anom_start=18, drop_depth=0.20),
    dict(pid="PT0004", name="Carlos Mendez",      age=68, sex="M",
         cancer_type="MM",        chemo_protocol="VAD",
         anomaly_type="sudden_drop", n_draws=26, anom_start=13, drop_depth=0.15),
    # sustained_elevation x4
    dict(pid="PT0005", name="Linda Reyes",        age=48, sex="F",
         cancer_type="Breast-CA", chemo_protocol="AC-T",
         anomaly_type="sustained_elevation", n_draws=30, anom_start=15, elev_factor=1.9),
    dict(pid="PT0006", name="James Whitfield",    age=71, sex="M",
         cancer_type="CLL",       chemo_protocol="FOLFOX",
         anomaly_type="sustained_elevation", n_draws=28, anom_start=14, elev_factor=1.5),
    dict(pid="PT0007", name="Amara Osei",          age=52, sex="F",
         cancer_type="AML",       chemo_protocol="VAD",
         anomaly_type="sustained_elevation", n_draws=30, anom_start=16, elev_factor=2.1),
    dict(pid="PT0008", name="Henrik Larsson",      age=60, sex="M",
         cancer_type="NHL",       chemo_protocol="R-CHOP",
         anomaly_type="sustained_elevation", n_draws=26, anom_start=13, elev_factor=1.6),
    # erratic x4
    dict(pid="PT0009", name="Susan Nakamura",      age=59, sex="F",
         cancer_type="MM",        chemo_protocol="VAD",
         anomaly_type="erratic",  n_draws=30, anom_start=14),
    dict(pid="PT0010", name="Robert Achebe",       age=44, sex="M",
         cancer_type="AML",       chemo_protocol="BEP",
         anomaly_type="erratic",  n_draws=28, anom_start=12),
    dict(pid="PT0011", name="Fatima Al-Rashid",    age=38, sex="F",
         cancer_type="NHL",       chemo_protocol="R-CHOP",
         anomaly_type="erratic",  n_draws=32, anom_start=17),
    dict(pid="PT0012", name="George Papadopoulos", age=66, sex="M",
         cancer_type="CLL",       chemo_protocol="FOLFOX",
         anomaly_type="erratic",  n_draws=26, anom_start=13),
    # none x4
    dict(pid="PT0013", name="Catherine Dubois",    age=53, sex="F",
         cancer_type="NHL",       chemo_protocol="R-CHOP",
         anomaly_type="none",     n_draws=30, nadir_depth=0.60),
    dict(pid="PT0014", name="Thomas Bergström",    age=67, sex="M",
         cancer_type="CLL",       chemo_protocol="FOLFOX",
         anomaly_type="none",     n_draws=28, nadir_depth=0.35),
    dict(pid="PT0015", name="Priya Sharma",         age=34, sex="F",
         cancer_type="Breast-CA", chemo_protocol="AC-T",
         anomaly_type="none",     n_draws=30, nadir_depth=0.50),
    dict(pid="PT0016", name="William Okonkwo",      age=74, sex="M",
         cancer_type="MM",        chemo_protocol="BEP",
         anomaly_type="none",     n_draws=26, nadir_depth=0.68),
]

def main():
    Path(PATIENTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    db = {}
    if Path(PATIENTS_FILE).exists():
        with open(PATIENTS_FILE) as f:
            db = json.load(f)

    old = [k for k in db if k.startswith("PT0")]
    for k in old:
        del db[k]
    if old:
        print(f"Removed {len(old)} old demo patient(s).")

    counts = {}
    for p in DEMO_PATIENTS:
        pid   = p["pid"]
        atype = p["anomaly_type"]
        db[pid] = build_patient(**p)
        counts[atype] = counts.get(atype,0)+1
        n = len(db[pid]["draws"])
        last_day = db[pid]["draws"][-1]["timestamp_day"]
        print(f"  {pid}  {p['name']:<24}  {atype:<22}  "
              f"{n} draws  ({last_day} days)")

    with open(PATIENTS_FILE,"w") as f:
        json.dump(db, f, indent=2)

    print(f"\n{len(DEMO_PATIENTS)} patients written → {PATIENTS_FILE}")
    for atype, c in sorted(counts.items()):
        print(f"  {'█'*c}  {c}x {atype}")

if __name__ == "__main__":
    main()

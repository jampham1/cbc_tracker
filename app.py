"""
CBC Anomaly Analyzer — Clinical Dashboard v2
=============================================
Redesigned app with:
  - Tab 1: Patient Management (create new / update existing patients)
  - Tab 2: Analytics (per-patient anomaly plots with anomaly type labels)

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBCTracker",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Syne+Mono&family=DM+Sans:wght@300;400;500&display=swap');

*, html, body { font-family: 'DM Sans', sans-serif; }
.block-container { padding: 1.8rem 2.4rem 2rem; max-width: 1400px; }

.cbc-header {
    display: flex; align-items: flex-end; gap: 1.2rem;
    border-bottom: 2px solid #1a2744;
    padding-bottom: 1rem; margin-bottom: 1.8rem;
}
.cbc-title {
    font-family: 'Syne', sans-serif; font-size: 1.85rem;
    font-weight: 700; color: #e8edf5; margin: 0; letter-spacing: -0.5px;
}
.cbc-badge {
    font-family: 'Syne Mono', monospace; font-size: 0.7rem; color: #4a9eff;
    background: rgba(74,158,255,0.1); border: 1px solid rgba(74,158,255,0.25);
    border-radius: 4px; padding: 3px 8px; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 2px solid #1a2744;
    background: transparent; margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif; font-size: 0.88rem; font-weight: 500;
    color: #6b7a99; background: transparent; border: none;
    border-bottom: 2px solid transparent; padding: 0.7rem 1.4rem;
    margin-bottom: -2px; transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #4a9eff !important; border-bottom: 2px solid #4a9eff !important;
    background: transparent !important;
}

.pt-card {
    background: #0d1628; border: 1px solid #1a2744; border-radius: 12px;
    padding: 1.1rem 1.3rem; margin-bottom: 0.7rem;
}
.pt-name { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.95rem; color: #c8d4e8; }
.pt-meta { font-family: 'Syne Mono', monospace; font-size: 0.72rem; color: #4a5a7a; margin-top: 3px; }
.pt-risk-high   { color: #f87171; font-weight: 600; }
.pt-risk-medium { color: #fbbf24; font-weight: 600; }
.pt-risk-low    { color: #34d399; font-weight: 600; }

.sec-label {
    font-family: 'Syne Mono', monospace; font-size: 0.68rem; color: #4a9eff;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.5rem; margin-top: 1.2rem;
}

.metric-strip { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-tile {
    background: #0d1628; border: 1px solid #1a2744; border-radius: 10px;
    padding: 0.8rem 1.1rem; flex: 1; min-width: 100px; text-align: center;
}
.metric-tile-label {
    font-family: 'Syne Mono', monospace; font-size: 0.65rem; color: #4a5a7a;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}
.metric-tile-value { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #c8d4e8; }
.metric-tile-sub   { font-size: 0.7rem; color: #4a5a7a; margin-top: 2px; }

.anom-badge {
    display: inline-block; font-family: 'Syne Mono', monospace;
    font-size: 0.7rem; font-weight: 500; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 0.06em; text-transform: uppercase;
}
.anom-sudden_drop         { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.anom-sustained_elevation { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.anom-erratic             { background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.anom-none                { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.anom-unknown             { background: rgba(100,116,139,0.12); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

.stNumberInput label, .stSelectbox label, .stTextInput label {
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.72rem !important; color: #6b7a99 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stForm"] {
    background: #0d1628; border: 1px solid #1a2744; border-radius: 12px; padding: 1.2rem;
}

.alert-danger {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.8rem 0;
    font-size: 0.88rem; color: #fca5a5;
}
.alert-ok {
    background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.2);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.8rem 0;
    font-size: 0.88rem; color: #6ee7b7;
}

[data-testid="stSidebar"] { background: #080f1e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CBC_FEATURES = ["WBC", "ANC", "RBC", "HGB", "PLT", "LYM", "MONO"]
UNITS = {
    "WBC": "×10⁹/L", "ANC": "×10⁹/L", "RBC": "×10¹²/L",
    "HGB": "g/dL",   "PLT": "×10⁹/L", "LYM": "×10⁹/L", "MONO": "×10⁹/L",
}
NORMAL_RANGES = {
    "WBC": (4.5, 11.0), "ANC": (1.8, 7.7),  "RBC": (4.2, 5.4),
    "HGB": (12.0, 17.5),"PLT": (150, 400),   "LYM": (1.0, 4.8),
    "MONO": (0.2, 0.95),
}
FEATURE_COLORS = {
    "WBC": "#4a9eff", "ANC": "#34d399", "RBC": "#fbbf24",
    "HGB": "#f87171", "PLT": "#a78bfa", "LYM": "#67e8f9", "MONO": "#f9a8d4",
}
CANCER_TYPES  = ["AML", "CLL", "NHL", "MM", "Breast-CA"]
CHEMO_PROTOS  = ["R-CHOP", "FOLFOX", "BEP", "VAD", "AC-T"]
ANOMALY_LABELS = {
    "sudden_drop":         "Sudden drop",
    "sustained_elevation": "Sustained elevation",
    "erratic":             "Erratic pattern",
    "none":                "No anomaly",
    "unknown":             "Unscored",
}
PATIENTS_FILE = "data/patients_db.json"

# ── Patient database ──────────────────────────────────────────────────────────
def load_patients() -> dict:
    Path(PATIENTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(PATIENTS_FILE).exists():
        with open(PATIENTS_FILE) as f:
            return json.load(f)
    return {}

def save_patients(db: dict):
    with open(PATIENTS_FILE, "w") as f:
        json.dump(db, f, indent=2)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        from predict import load_inference_artifacts
        return load_inference_artifacts(
            ckpt_path         = "outputs/checkpoints/best_model.pt",
            preprocessor_path = "outputs/preprocessor.pkl",
        )
    except Exception:
        return None, None, None

# ── Anomaly type classifier ───────────────────────────────────────────────────
def classify_anomaly_type(scores: np.ndarray, threshold: float = 0.5) -> str:
    flagged = (scores >= threshold).astype(int)
    if flagged.sum() == 0:
        return "none"
    runs, current = [], 0
    for f in flagged:
        if f == 1:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    if not runs:
        return "none"
    if len(runs) >= 3:
        return "erratic"
    if max(runs) >= 5:
        return "sustained_elevation"
    return "sudden_drop"

# ── Score patient ─────────────────────────────────────────────────────────────
def score_patient(pid: str, db: dict, model, preprocessor, device):
    from predict import predict_patient
    draws = db[pid]["draws"]
    if len(draws) < 15:
        return None
    patient_df = pd.DataFrame(draws).sort_values("timestamp_day").reset_index(drop=True)
    patient_df["is_anomaly"]   = 0
    patient_df["anomaly_type"] = "none"
    try:
        return predict_patient(patient_df, model, preprocessor, device)
    except Exception:
        return None

# ── Chart ─────────────────────────────────────────────────────────────────────
def build_chart(patient_df, pred_df, feature, anomaly_type):
    lo, hi = NORMAL_RANGES[feature]
    color  = FEATURE_COLORS[feature]
    fig    = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(52,211,153,0.05)", line_width=0, secondary_y=False)
    fig.add_hline(y=lo, line_dash="dot", line_color="rgba(52,211,153,0.3)", line_width=1, secondary_y=False)
    fig.add_hline(y=hi, line_dash="dot", line_color="rgba(52,211,153,0.3)", line_width=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=patient_df["timestamp_day"], y=patient_df[feature],
        name=f"{feature} ({UNITS[feature]})",
        line=dict(color=color, width=2.5), mode="lines+markers",
        marker=dict(size=5, color=color, line=dict(color="#080f1e", width=1.5)),
        hovertemplate=f"<b>Day %{{x}}</b><br>{feature}: %{{y:.2f}} {UNITS[feature]}<extra></extra>",
    ), secondary_y=False)

    if pred_df is not None and "anomaly_score" in pred_df.columns:
        scored = pred_df.dropna(subset=["anomaly_score"])
        if len(scored):
            mc = scored["anomaly_score"].apply(
                lambda s: "#f87171" if s >= 0.7 else "#fbbf24" if s >= 0.4 else "#34d399"
            )
            fig.add_trace(go.Scatter(
                x=scored["timestamp_day"], y=scored["anomaly_score"],
                name="Anomaly score",
                line=dict(color="rgba(248,113,113,0.85)", width=2),
                mode="lines+markers",
                marker=dict(size=6, color=mc, line=dict(color="#080f1e", width=1)),
                fill="tozeroy", fillcolor="rgba(248,113,113,0.05)",
                hovertemplate="<b>Day %{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ), secondary_y=True)
            fig.add_hline(y=0.5, line_dash="dash",
                          line_color="rgba(248,113,113,0.35)", line_width=1, secondary_y=True)

    anom_color = {
        "sudden_drop": "rgba(248,113,113,0.8)",
        "sustained_elevation": "rgba(251,191,36,0.8)",
        "erratic": "rgba(167,139,250,0.8)", "none": "rgba(52,211,153,0.8)",
    }.get(anomaly_type, "rgba(148,163,184,0.8)")

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text=f"⬤  {ANOMALY_LABELS.get(anomaly_type, anomaly_type)}",
        showarrow=False, font=dict(size=11, color=anom_color, family="Syne Mono"),
        align="left",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(family="DM Sans", color="#6b7a99"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=8, r=8, t=36, b=36), hovermode="x unified", height=340,
        xaxis=dict(title="Days from first draw",
                   gridcolor="rgba(255,255,255,0.04)",
                   showline=True, linecolor="rgba(255,255,255,0.08)"),
    )
    fig.update_yaxes(title_text=f"{feature} ({UNITS[feature]})",
                     gridcolor="rgba(255,255,255,0.04)", secondary_y=False,
                     title_font=dict(color=color))
    fig.update_yaxes(title_text="Anomaly score", range=[0, 1.05],
                     secondary_y=True, gridcolor="rgba(0,0,0,0)",
                     title_font=dict(color="#f87171"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="cbc-header">
        <h1 class="cbc-title"> CBCTracker </h1>
        <span class="cbc-badge">Clinical Decision Support</span>
    </div>
    """, unsafe_allow_html=True)

    model, preprocessor, device = load_model()
    db = load_patients()

    if model is None:
        st.error("Model not loaded. Run `train.py` first.")
        return

    if "selected_pid" not in st.session_state:
        st.session_state.selected_pid = None

    tab1, tab2 = st.tabs(["👤  Patient Management", "📊  Analytics"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Patient Management
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown('<p class="sec-label">Action</p>', unsafe_allow_html=True)
            action = st.radio(
                "action",
                ["Create new patient", "Update existing patient"],
                label_visibility="collapsed",
            )

            if action == "Update existing patient":
                if db:
                    st.markdown('<p class="sec-label">Select Patient</p>',
                                unsafe_allow_html=True)
                    opts = {pid: f"{info['name']}  ({pid})"
                            for pid, info in db.items()}
                    chosen = st.selectbox("patient", list(opts.values()),
                                          label_visibility="collapsed")
                    st.session_state.selected_pid = [k for k, v in opts.items()
                                                      if v == chosen][0]
                else:
                    st.info("No patients yet — create one first.")

            # Roster
            if db:
                st.markdown('<p class="sec-label">Patient Roster</p>',
                            unsafe_allow_html=True)
                for pid, info in list(db.items())[-10:]:
                    n    = len(info.get("draws", []))
                    ps   = info.get("last_peak_score")
                    rc   = ("pt-risk-high"   if ps and ps >= 0.7 else
                            "pt-risk-medium" if ps and ps >= 0.4 else "pt-risk-low")
                    risk = (f"<span class='{rc}'>● {ps:.2f}</span>"
                            if ps else "<span style='color:#2a3a5a'>● unscored</span>")
                    st.markdown(f"""
                    <div class="pt-card">
                        <div class="pt-name">{info['name']}</div>
                        <div class="pt-meta">
                            {info.get('cancer_type','—')} &nbsp;·&nbsp;
                            {n} draws &nbsp;·&nbsp; {risk}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with right:
            # ── CREATE ────────────────────────────────────────────────────────
            if action == "Create new patient":
                st.markdown('<p class="sec-label">New Patient Details</p>',
                            unsafe_allow_html=True)
                with st.form("create_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        nm  = st.text_input("Patient name")
                        age = st.number_input("Age", 18, 95, 55)
                        sex = st.selectbox("Sex", ["M", "F"])
                    with c2:
                        can = st.selectbox("Cancer type", CANCER_TYPES)
                        chm = st.selectbox("Chemo protocol", CHEMO_PROTOS)
                        cid = st.text_input("Patient ID (leave blank to auto-generate)")

                    st.markdown('<p class="sec-label">First CBC Draw</p>',
                                unsafe_allow_html=True)
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        wbc  = st.number_input("WBC",  0.0, 100.0, 7.0,  0.1)
                        anc  = st.number_input("ANC",  0.0,  50.0, 3.5,  0.1)
                    with f2:
                        rbc  = st.number_input("RBC",  0.0,  10.0, 4.8,  0.1)
                        hgb  = st.number_input("HGB",  0.0,  25.0,14.0,  0.1)
                    with f3:
                        plt_ = st.number_input("PLT",  0.0,1000.0,250.0, 10.0)
                        lym  = st.number_input("LYM",  0.0,  20.0, 2.0,  0.1)
                    with f4:
                        mono = st.number_input("MONO", 0.0,   5.0, 0.5,  0.05)
                        day  = st.number_input("Day from diagnosis", 0, 9999, 0)

                    ok = st.form_submit_button("✚  Create Patient",
                                               use_container_width=True)

                if ok:
                    if not nm.strip():
                        st.error("Patient name is required.")
                    else:
                        pid = cid.strip() if cid.strip() else f"PT{len(db)+1:04d}"
                        db[pid] = {
                            "name": nm.strip(), "age": int(age), "sex": sex,
                            "cancer_type": can, "chemo_protocol": chm,
                            "created_at": datetime.now().isoformat(),
                            "draws": [{
                                "patient_id": pid, "timestamp_day": int(day),
                                "time_delta": 0,
                                "WBC": wbc, "ANC": anc, "RBC": rbc,
                                "HGB": hgb, "PLT": plt_, "LYM": lym, "MONO": mono,
                                "age": int(age), "sex": sex,
                                "cancer_type": can, "chemo_protocol": chm,
                            }],
                        }
                        save_patients(db)
                        st.session_state.selected_pid = pid
                        st.success(f"✓ Patient '{nm.strip()}' created (ID: {pid})")
                        st.rerun()

            # ── UPDATE ────────────────────────────────────────────────────────
            else:
                pid = st.session_state.selected_pid
                if pid and pid in db:
                    info  = db[pid]
                    draws = info["draws"]
                    last  = draws[-1]["timestamp_day"] if draws else 0

                    st.markdown(f"""
                    <div class="metric-strip">
                        <div class="metric-tile">
                            <div class="metric-tile-label">Name</div>
                            <div class="metric-tile-value" style="font-size:1rem">{info['name']}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">ID</div>
                            <div class="metric-tile-value" style="font-size:1rem">{pid}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Cancer</div>
                            <div class="metric-tile-value" style="font-size:1rem">{info.get('cancer_type','—')}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Draws</div>
                            <div class="metric-tile-value">{len(draws)}</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Last draw</div>
                            <div class="metric-tile-value" style="font-size:1rem">Day {last}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if len(draws) < 15:
                        st.markdown(
                            f'<div class="alert-danger">⚠ '
                            f'{15-len(draws)} more draw(s) needed before scoring.</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<p class="sec-label">Add New CBC Draw</p>',
                                unsafe_allow_html=True)

                    with st.form(f"add_{pid}"):
                        u1, u2, u3, u4 = st.columns(4)
                        with u1:
                            a_wbc  = st.number_input("WBC",  0.0, 100.0, 7.0,  0.1)
                            a_anc  = st.number_input("ANC",  0.0,  50.0, 3.5,  0.1)
                        with u2:
                            a_rbc  = st.number_input("RBC",  0.0,  10.0, 4.8,  0.1)
                            a_hgb  = st.number_input("HGB",  0.0,  25.0,14.0,  0.1)
                        with u3:
                            a_plt  = st.number_input("PLT",  0.0,1000.0,250.0,10.0)
                            a_lym  = st.number_input("LYM",  0.0,  20.0, 2.0,  0.1)
                        with u4:
                            a_mono = st.number_input("MONO", 0.0,   5.0, 0.5,  0.05)
                            a_day  = st.number_input(
                                "Day from diagnosis",
                                min_value=last, value=last + 7, step=1,
                            )
                        add_ok = st.form_submit_button("➕  Add Draw",
                                                        use_container_width=True)

                    if add_ok:
                        db[pid]["draws"].append({
                            "patient_id": pid,
                            "timestamp_day": int(a_day),
                            "time_delta": int(a_day) - last,
                            "WBC": a_wbc, "ANC": a_anc, "RBC": a_rbc,
                            "HGB": a_hgb, "PLT": a_plt, "LYM": a_lym,
                            "MONO": a_mono,
                            "age": info["age"], "sex": info["sex"],
                            "cancer_type": info["cancer_type"],
                            "chemo_protocol": info["chemo_protocol"],
                        })
                        save_patients(db)
                        st.success(f"✓ Draw at day {int(a_day)} added.")
                        st.rerun()

                    with st.expander("Draw history & manage draws",
                                     expanded=False):
                        if draws:
                            hist = pd.DataFrame(draws)[
                                ["timestamp_day"] + CBC_FEATURES
                                ].round(2)
                            st.dataframe(hist, use_container_width=True,
                                         height=200)

                            st.markdown(
                                '<p class="sec-label">Delete a draw</p>',
                                unsafe_allow_html=True
                            )
                            draw_options = {
                                i: f"Draw {i+1}  —  Day {d['timestamp_day']}  "
                                   f"(WBC {d['WBC']:.1f}, ANC {d['ANC']:.1f})"
                                for i, d in enumerate(draws)
                            }
                            del_idx = st.selectbox(
                                "Select draw to delete",
                                list(draw_options.keys()),
                                format_func=lambda i: draw_options[i],
                                key=f"del_draw_{pid}",
                            )
                            if st.button("🗑  Delete selected draw",
                                         key=f"del_draw_btn_{pid}",
                                         use_container_width=True):
                                db[pid]["draws"].pop(del_idx)
                                # Recompute time_deltas after deletion
                                for k in range(len(db[pid]["draws"])):
                                    if k == 0:
                                        db[pid]["draws"][k]["time_delta"] = 0
                                    else:
                                        db[pid]["draws"][k]["time_delta"] = (
                                            db[pid]["draws"][k]["timestamp_day"]
                                            - db[pid]["draws"][k-1]["timestamp_day"]
                                        )
                                save_patients(db)
                                st.success(
                                    f"✓ Draw at day "
                                    f"{draw_options[del_idx].split('Day')[1].split()[0]}"
                                    f" deleted."
                                )
                                st.rerun()
                        else:
                            st.info("No draws on record.")

                    # ── Danger zone: delete patient ───────────────────────
                    with st.expander("⚠  Danger zone", expanded=False):
                        st.markdown(
                            '<p style="color:#f87171;font-size:0.85rem">'
                            'Permanently deletes this patient and all their '
                            'CBC draws. This cannot be undone.</p>',
                            unsafe_allow_html=True
                        )
                        confirm = st.text_input(
                            f"Type  {pid}  to confirm deletion",
                            key=f"confirm_del_{pid}",
                            placeholder=f"Type {pid} here...",
                        )
                        if st.button("🗑  Delete patient permanently",
                                     key=f"del_patient_btn_{pid}",
                                     use_container_width=True):
                            if confirm.strip() == pid:
                                del db[pid]
                                save_patients(db)
                                st.session_state.selected_pid = None
                                st.success("Patient deleted.")
                                st.rerun()
                            else:
                                st.error(
                                    f"ID doesn't match. "
                                    f"Type exactly  {pid}  to confirm."
                                )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Analytics
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        if not db:
            st.markdown("""
            <div style="text-align:center;padding:4rem;color:#2a3a5a;
                        border:1px dashed #1a2744;border-radius:12px">
                <p style="font-size:2.5rem;margin:0">📋</p>
                <p style="font-family:'Syne Mono',monospace;font-size:0.8rem;
                          letter-spacing:0.1em;text-transform:uppercase;margin-top:0.8rem">
                    No patients yet — add one in Patient Management
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Controls row
        ac1, ac2, ac3 = st.columns([2, 1, 1])
        with ac1:
            opts    = {pid: f"{info['name']}  ({pid})" for pid, info in db.items()}
            chosen  = st.selectbox("Select patient", list(opts.values()))
            vp      = [k for k, v in opts.items() if v == chosen][0]
        with ac2:
            feat    = st.selectbox("CBC overlay", CBC_FEATURES,
                                   format_func=lambda f: f"{f} ({UNITS[f]})")
        with ac3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶  Run scoring", use_container_width=True)

        info  = db[vp]
        draws = info["draws"]

        if len(draws) < 15:
            st.markdown(
                f'<div class="alert-danger">⚠ Only {len(draws)} draws on record. '
                f'Minimum 15 required for scoring.</div>',
                unsafe_allow_html=True
            )
            return

        patient_df = pd.DataFrame(draws).sort_values("timestamp_day").reset_index(drop=True)
        patient_df["is_anomaly"]   = 0
        patient_df["anomaly_type"] = "none"

        pred_df      = None
        anomaly_type = info.get("last_anomaly_type", "unknown")
        peak_score   = info.get("last_peak_score",   None)

        if run_btn:
            with st.spinner("Scoring patient timeline..."):
                pred_df = score_patient(vp, db, model, preprocessor, device)
            if pred_df is not None and "anomaly_score" in pred_df.columns:
                scores       = pred_df["anomaly_score"].dropna().values
                peak_score   = float(scores.max()) if len(scores) else 0.0
                anomaly_type = classify_anomaly_type(scores)
                db[vp]["last_peak_score"]   = peak_score
                db[vp]["last_anomaly_type"] = anomaly_type
                save_patients(db)

        # Info strip
        rc  = ("pt-risk-high"   if peak_score and peak_score >= 0.7 else
               "pt-risk-medium" if peak_score and peak_score >= 0.4 else "pt-risk-low")
        rv  = f"{peak_score:.3f}" if peak_score is not None else "—"
        acs = anomaly_type.replace(" ", "_")

        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric-tile">
                <div class="metric-tile-label">Patient</div>
                <div class="metric-tile-value" style="font-size:1rem">{info['name']}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Age / Sex</div>
                <div class="metric-tile-value" style="font-size:1rem">
                    {info.get('age','—')} / {info.get('sex','—')}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Cancer</div>
                <div class="metric-tile-value" style="font-size:1rem">
                    {info.get('cancer_type','—')}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Protocol</div>
                <div class="metric-tile-value" style="font-size:1rem">
                    {info.get('chemo_protocol','—')}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Draws</div>
                <div class="metric-tile-value">{len(draws)}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Peak score</div>
                <div class="metric-tile-value {rc}">{rv}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Pattern</div>
                <div class="metric-tile-value" style="font-size:0.85rem">
                    <span class="anom-badge anom-{acs}">
                        {ANOMALY_LABELS.get(anomaly_type, anomaly_type)}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Alert banner
        if pred_df is not None and "anomaly_score" in pred_df.columns:
            hr = pred_df[pred_df["anomaly_score"] >= 0.7]
            if len(hr):
                days = [int(d) for d in hr["timestamp_day"].tolist()[:8]]
                st.markdown(
                    f'<div class="alert-danger">⚠ <strong>{len(hr)} '
                    f'high-risk draw(s)</strong> detected — days {days}'
                    f'{"…" if len(hr) > 8 else ""}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alert-ok">✓ No high-risk draws at threshold 0.70</div>',
                    unsafe_allow_html=True
                )

        # Merge pred into patient_df for chart
        if pred_df is not None:
            merged = patient_df.merge(
                pred_df[["timestamp_day", "anomaly_score", "risk_level"]],
                on="timestamp_day", how="left"
            )
        else:
            merged = patient_df.copy()

        # Main chart
        st.markdown(f'<p class="sec-label">{feat} vs anomaly score</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            build_chart(merged, merged if pred_df is not None else None,
                        feat, anomaly_type),
            use_container_width=True,
        )

        # All CBC mini grid
        with st.expander("All CBC features", expanded=False):
            rows = [CBC_FEATURES[i:i+4] for i in range(0, len(CBC_FEATURES), 4)]
            for row_feats in rows:
                cols = st.columns(len(row_feats))
                for col, f in zip(cols, row_feats):
                    lo, hi = NORMAL_RANGES[f]
                    mf = go.Figure()
                    mf.add_hrect(y0=lo, y1=hi,
                                 fillcolor="rgba(52,211,153,0.05)", line_width=0)
                    mf.add_trace(go.Scatter(
                        x=patient_df["timestamp_day"], y=patient_df[f],
                        line=dict(color=FEATURE_COLORS[f], width=2),
                        mode="lines+markers", marker=dict(size=3),
                        showlegend=False,
                    ))
                    mf.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.015)",
                        margin=dict(l=4, r=4, t=28, b=24), height=170,
                        title=dict(text=f, font=dict(
                            size=11, color=FEATURE_COLORS[f], family="Syne Mono"
                        )),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                                   tickfont=dict(size=8)),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                                   tickfont=dict(size=8)),
                    )
                    col.plotly_chart(mf, use_container_width=True)

        # Download
        st.markdown('<p class="sec-label">Export</p>', unsafe_allow_html=True)
        export = merged.copy()
        if "anomaly_score" not in export.columns:
            export["anomaly_score"] = np.nan
            export["risk_level"]    = "unscored"
        export["detected_anomaly_type"] = anomaly_type
        export["patient_name"]          = info["name"]

        st.download_button(
            label    = "⬇  Download full patient report (CSV)",
            data     = export.to_csv(index=False).encode(),
            file_name= f"{info['name'].replace(' ','_')}_{vp}_cbc_report.csv",
            mime     = "text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
"""
CBC Anomaly Analyzer — Clinical Dashboard
==========================================
Streamlit app for oncologists to:
  1. Browse existing patients with interactive CBC + anomaly overlays
  2. Submit new patient CBC data and get real-time anomaly scores

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBC Anomaly Analyzer",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main { background: #0a0e1a; }
    .block-container { padding: 2rem 2.5rem; }

    /* Header */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(99,179,237,0.2);
        padding-bottom: 1.2rem;
    }
    .app-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #63b3ed;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'DM Mono', monospace;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #e2e8f0;
        line-height: 1;
    }
    .metric-unit {
        font-size: 0.75rem;
        color: #718096;
        margin-top: 0.2rem;
    }

    /* Risk badge */
    .risk-high   { color: #fc8181; font-weight: 600; }
    .risk-medium { color: #f6ad55; font-weight: 600; }
    .risk-low    { color: #68d391; font-weight: 600; }

    /* Section headers */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #63b3ed;
        margin-bottom: 0.5rem;
    }

    /* Input form styling */
    .stNumberInput label, .stSelectbox label, .stTextInput label {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid rgba(99,179,237,0.1);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #a0aec0 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 8px;
        color: #a0aec0;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,179,237,0.12) !important;
        border-color: #63b3ed !important;
        color: #63b3ed !important;
    }

    /* Alert box */
    .alert-box {
        background: rgba(252,129,129,0.08);
        border: 1px solid rgba(252,129,129,0.3);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .alert-box-low {
        background: rgba(104,211,145,0.08);
        border: 1px solid rgba(104,211,145,0.3);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Clinical normal ranges ────────────────────────────────────────────────────
NORMAL_RANGES = {
    "WBC":  (4.5,  11.0),
    "ANC":  (1.8,   7.7),
    "RBC":  (4.2,   5.4),
    "HGB":  (12.0, 17.5),
    "PLT":  (150,  400),
    "LYM":  (1.0,   4.8),
    "MONO": (0.2,   0.95),
}
CBC_FEATURES = list(NORMAL_RANGES.keys())
UNITS = {
    "WBC": "×10⁹/L", "ANC": "×10⁹/L", "RBC": "×10¹²/L",
    "HGB": "g/dL",   "PLT": "×10⁹/L", "LYM": "×10⁹/L", "MONO": "×10⁹/L"
}
FEATURE_COLORS = {
    "WBC":  "#63b3ed", "ANC":  "#68d391", "RBC":  "#f6ad55",
    "HGB":  "#fc8181", "PLT":  "#b794f4", "LYM":  "#76e4f7", "MONO": "#fbb6ce",
}

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    try:
        from predict import load_inference_artifacts
        model, preprocessor, device = load_inference_artifacts(
            ckpt_path         = "outputs/checkpoints/best_model.pt",
            preprocessor_path = "outputs/preprocessor.pkl",
        )
        return model, preprocessor, device
    except Exception as e:
        return None, None, None

@st.cache_data
def load_patient_data():
    try:
        return pd.read_csv("data/cbc_synthetic.csv")
    except:
        return None

def get_patient_predictions(patient_df, model, preprocessor, device):
    from predict import predict_patient
    return predict_patient(patient_df, model, preprocessor, device)

# ── Chart builder ─────────────────────────────────────────────────────────────
def build_overlay_chart(patient_df: pd.DataFrame, pred_df: pd.DataFrame, feature: str):
    """
    Build a dual-axis Plotly chart: CBC feature on left, anomaly score on right.
    Shades the true anomaly window and marks clinical normal range.
    """
    lo, hi = NORMAL_RANGES[feature]
    color  = FEATURE_COLORS[feature]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Normal range band ─────────────────────────────────────────────────────
    fig.add_hrect(
        y0=lo, y1=hi,
        fillcolor="rgba(104,211,145,0.07)",
        line_width=0,
        secondary_y=False,
    )
    fig.add_hline(y=lo, line_dash="dot", line_color="rgba(104,211,145,0.4)",
                  line_width=1, secondary_y=False)
    fig.add_hline(y=hi, line_dash="dot", line_color="rgba(104,211,145,0.4)",
                  line_width=1, secondary_y=False)

    # ── True anomaly window shading ───────────────────────────────────────────
    if "is_anomaly" in patient_df.columns:
        anom = patient_df[patient_df["is_anomaly"] == 1]
        if len(anom) > 0:
            fig.add_vrect(
                x0=anom["timestamp_day"].min(),
                x1=anom["timestamp_day"].max(),
                fillcolor="rgba(252,129,129,0.08)",
                line_color="rgba(252,129,129,0.3)",
                line_width=1,
                annotation_text="Anomaly window",
                annotation_position="top left",
                annotation_font_color="rgba(252,129,129,0.7)",
                annotation_font_size=11,
            )

    # ── CBC feature line ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=patient_df["timestamp_day"],
            y=patient_df[feature],
            name=f"{feature} ({UNITS[feature]})",
            line=dict(color=color, width=2.5),
            mode="lines+markers",
            marker=dict(size=5, color=color,
                        line=dict(color="#0a0e1a", width=1.5)),
            hovertemplate=(
                f"<b>Day %{{x}}</b><br>"
                f"{feature}: %{{y:.2f}} {UNITS[feature]}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # ── Anomaly score line ────────────────────────────────────────────────────
    if pred_df is not None and "anomaly_score" in pred_df.columns:
        scored = pred_df.dropna(subset=["anomaly_score"])
        if len(scored) > 0:
            # Color-coded markers by risk level
            marker_colors = scored["anomaly_score"].apply(
                lambda s: "#fc8181" if s >= 0.7
                          else "#f6ad55" if s >= 0.4
                          else "#68d391"
            )
            fig.add_trace(
                go.Scatter(
                    x=scored["timestamp_day"],
                    y=scored["anomaly_score"],
                    name="Anomaly score",
                    line=dict(color="rgba(252,129,129,0.9)", width=2),
                    mode="lines+markers",
                    marker=dict(size=7, color=marker_colors,
                                line=dict(color="#0a0e1a", width=1)),
                    hovertemplate=(
                        "<b>Day %{x}</b><br>"
                        "Anomaly score: %{y:.3f}<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

            # Threshold line
            fig.add_hline(
                y=0.5, line_dash="dash",
                line_color="rgba(252,129,129,0.4)",
                line_width=1,
                secondary_y=True,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Outfit, sans-serif", color="#a0aec0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=40),
        hovermode="x unified",
        xaxis=dict(
            title="Days from diagnosis",
            gridcolor="rgba(255,255,255,0.05)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
            tickfont=dict(size=11),
        ),
        height=380,
    )
    fig.update_yaxes(
        title_text=f"{feature}  ({UNITS[feature]})",
        gridcolor="rgba(255,255,255,0.05)",
        secondary_y=False,
        tickfont=dict(size=11),
        title_font=dict(color=color),
    )
    fig.update_yaxes(
        title_text="Anomaly score",
        range=[0, 1],
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11),
        title_font=dict(color="#fc8181"),
    )

    return fig


def build_all_features_chart(patient_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Mini sparkline grid showing all 7 features + score."""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=CBC_FEATURES + ["Anomaly score"],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]

    for i, feat in enumerate(CBC_FEATURES):
        r, c = positions[i]
        lo, hi = NORMAL_RANGES[feat]
        color  = FEATURE_COLORS[feat]

        fig.add_hrect(y0=lo, y1=hi,
                      fillcolor="rgba(104,211,145,0.06)",
                      line_width=0, row=r, col=c)
        fig.add_trace(go.Scatter(
            x=patient_df["timestamp_day"], y=patient_df[feat],
            line=dict(color=color, width=1.8),
            mode="lines+markers", marker=dict(size=3),
            showlegend=False,
            hovertemplate=f"{feat}: %{{y:.2f}}<extra></extra>",
        ), row=r, col=c)

    # Anomaly score
    if pred_df is not None and "anomaly_score" in pred_df.columns:
        scored = pred_df.dropna(subset=["anomaly_score"])
        fig.add_trace(go.Scatter(
            x=scored["timestamp_day"], y=scored["anomaly_score"],
            fill="tozeroy",
            fillcolor="rgba(252,129,129,0.15)",
            line=dict(color="#fc8181", width=2),
            mode="lines", showlegend=False,
            hovertemplate="Score: %{y:.3f}<extra></extra>",
        ), row=2, col=4)
        fig.add_hline(y=0.5, line_dash="dash",
                      line_color="rgba(252,129,129,0.4)",
                      line_width=1, row=2, col=4)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Outfit, sans-serif", color="#a0aec0", size=10),
        margin=dict(l=5, r=5, t=30, b=5),
        height=340,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🩸 CBC Anomaly Analyzer</h1>
        <span class="app-subtitle">Oncology Clinical Decision Support</span>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    model, preprocessor, device = load_model_artifacts()
    df = load_patient_data()

    if model is None:
        st.error("⚠️ Could not load model. Run `train.py` first to generate "
                 "`outputs/checkpoints/best_model.pt`.")
        return

    if df is None:
        st.error("⚠️ Could not load patient data from `data/cbc_synthetic.csv`.")
        return

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊  Patient Explorer", "➕  New Patient"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Patient Explorer
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        col_sidebar, col_main = st.columns([1, 3], gap="large")

        with col_sidebar:
            st.markdown('<p class="section-label">Patient Selection</p>',
                        unsafe_allow_html=True)

            # Filter by anomaly type
            anom_filter = st.selectbox(
                "Filter by anomaly type",
                ["All"] + sorted(df["anomaly_type"].unique().tolist()),
            )
            filtered_df = df if anom_filter == "All" \
                          else df[df["anomaly_type"] == anom_filter]

            patient_ids = sorted(filtered_df["patient_id"].unique().tolist())
            selected_pid = st.selectbox(
                "Patient ID",
                patient_ids,
                format_func=lambda x: f"Patient {x:03d}",
            )

            st.markdown("---")
            st.markdown('<p class="section-label">CBC Feature Overlay</p>',
                        unsafe_allow_html=True)

            selected_feature = st.selectbox(
                "CBC variable to overlay",
                CBC_FEATURES,
                format_func=lambda f: f"{f}  ({UNITS[f]})",
            )

            show_overview = st.checkbox("Show all features overview", value=False)

        with col_main:
            # Load patient
            patient_df = df[df["patient_id"] == selected_pid]\
                           .sort_values("timestamp_day").reset_index(drop=True)

            # Run inference
            with st.spinner("Scoring patient timeline..."):
                pred_df = get_patient_predictions(
                    patient_df, model, preprocessor, device
                )

            # ── Patient info bar ──────────────────────────────────────────────
            info = patient_df.iloc[0]
            max_score = pred_df["anomaly_score"].max() \
                        if pred_df is not None else 0
            risk_color = ("risk-high"   if max_score >= 0.7 else
                          "risk-medium" if max_score >= 0.4 else "risk-low")
            risk_label = ("HIGH"   if max_score >= 0.7 else
                          "MEDIUM" if max_score >= 0.4 else "LOW")

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">Patient</div>
                    <div class="metric-value">{selected_pid:03d}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Age</div>
                    <div class="metric-value">{int(info['age'])}</div>
                    <div class="metric-unit">years</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Diagnosis</div>
                    <div class="metric-value" style="font-size:1.1rem">
                        {info['cancer_type']}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Protocol</div>
                    <div class="metric-value" style="font-size:1.1rem">
                        {info['chemo_protocol']}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CBC Draws</div>
                    <div class="metric-value">{len(patient_df)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Peak Risk</div>
                    <div class="metric-value {risk_color}">{risk_label}</div>
                    <div class="metric-unit">{max_score:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Alert banner ──────────────────────────────────────────────────
            high_risk_draws = pred_df[pred_df["anomaly_score"] >= 0.7] \
                              if pred_df is not None else pd.DataFrame()

            if len(high_risk_draws) > 0:
                days = high_risk_draws["timestamp_day"].tolist()
                st.markdown(f"""
                <div class="alert-box">
                    ⚠️ <strong style="color:#fc8181">
                    {len(high_risk_draws)} high-risk draws detected</strong>
                    — days {[int(d) for d in days[:8]]}
                    {'…' if len(days) > 8 else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box-low">
                    ✓ <strong style="color:#68d391">No high-risk draws detected
                    </strong> at threshold 0.70
                </div>
                """, unsafe_allow_html=True)

            # ── Main chart ────────────────────────────────────────────────────
            st.markdown(
                f'<p class="section-label">'
                f'{selected_feature} vs anomaly score</p>',
                unsafe_allow_html=True
            )
            fig = build_overlay_chart(patient_df, pred_df, selected_feature)
            st.plotly_chart(fig, use_container_width=True)

            # ── Overview sparklines ───────────────────────────────────────────
            if show_overview:
                st.markdown(
                    '<p class="section-label">All CBC features overview</p>',
                    unsafe_allow_html=True
                )
                fig_all = build_all_features_chart(patient_df, pred_df)
                st.plotly_chart(fig_all, use_container_width=True)

            # ── Raw data table ────────────────────────────────────────────────
            with st.expander("Raw draw data"):
                display_cols = ["timestamp_day", "time_delta"] + \
                               CBC_FEATURES + ["anomaly_score", "risk_level"]
                if pred_df is not None:
                    merged = patient_df.merge(
                        pred_df[["timestamp_day", "anomaly_score", "risk_level"]],
                        on="timestamp_day", how="left"
                    )
                    st.dataframe(
                        merged[display_cols].round(3),
                        use_container_width=True,
                        height=260,
                    )
                else:
                    st.dataframe(patient_df[display_cols[:-2]].round(3),
                                 use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — New Patient Input
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <p style="color:#a0aec0; margin-bottom:1.5rem">
        Enter a new patient's CBC draws to generate real-time anomaly scores.
        Add one draw at a time using the form below.
        </p>
        """, unsafe_allow_html=True)

        # ── Patient metadata ──────────────────────────────────────────────────
        st.markdown('<p class="section-label">Patient Information</p>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            new_age = st.number_input("Age", min_value=18, max_value=95,
                                      value=55, step=1)
        with m2:
            new_sex = st.selectbox("Sex", ["M", "F"])
        with m3:
            new_cancer = st.selectbox(
                "Cancer type",
                ["AML", "CLL", "NHL", "MM", "Breast-CA"]
            )
        with m4:
            new_chemo = st.selectbox(
                "Chemo protocol",
                ["R-CHOP", "FOLFOX", "BEP", "VAD", "AC-T"]
            )

        st.markdown("---")

        # ── Draw input form ───────────────────────────────────────────────────
        st.markdown('<p class="section-label">Add CBC Draw</p>',
                    unsafe_allow_html=True)

        with st.form("new_draw_form", clear_on_submit=True):
            day_col, _ = st.columns([1, 3])
            with day_col:
                draw_day = st.number_input(
                    "Day from diagnosis", min_value=0, value=0, step=1
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                wbc  = st.number_input("WBC  (×10⁹/L)",  0.0, 100.0, 7.0,  0.1)
                anc  = st.number_input("ANC  (×10⁹/L)",  0.0,  50.0, 3.5,  0.1)
            with c2:
                rbc  = st.number_input("RBC  (×10¹²/L)", 0.0,  10.0, 4.8,  0.1)
                hgb  = st.number_input("HGB  (g/dL)",     0.0,  25.0, 14.0, 0.1)
            with c3:
                plt_ = st.number_input("PLT  (×10⁹/L)",  0.0, 1000.0,250.0,10.0)
                lym  = st.number_input("LYM  (×10⁹/L)",  0.0,  20.0, 2.0,  0.1)
            with c4:
                mono = st.number_input("MONO (×10⁹/L)",  0.0,   5.0, 0.5,  0.05)

            submitted = st.form_submit_button(
                "➕  Add Draw",
                use_container_width=True,
            )

        # ── Session state: accumulate draws ──────────────────────────────────
        if "new_patient_draws" not in st.session_state:
            st.session_state.new_patient_draws = []

        if submitted:
            draws = st.session_state.new_patient_draws
            prev_day = draws[-1]["timestamp_day"] if draws else draw_day
            st.session_state.new_patient_draws.append({
                "patient_id":     9999,
                "timestamp_day":  draw_day,
                "time_delta":     draw_day - prev_day if draws else 0,
                "WBC":  wbc, "ANC": anc, "RBC": rbc, "HGB": hgb,
                "PLT":  plt_, "LYM": lym, "MONO": mono,
                "age":            new_age,
                "sex":            new_sex,
                "cancer_type":    new_cancer,
                "chemo_protocol": new_chemo,
                "is_anomaly":     0,
                "anomaly_type":   "none",
            })
            st.success(f"Draw at day {draw_day} added. "
                       f"Total draws: {len(st.session_state.new_patient_draws)}")

        # ── Controls ──────────────────────────────────────────────────────────
        btn1, btn2, btn3 = st.columns([1, 1, 3])
        with btn1:
            if st.button("🗑️  Clear all draws", use_container_width=True):
                st.session_state.new_patient_draws = []
                st.rerun()
        with btn2:
            if st.button("📥  Load example patient", use_container_width=True):
                # Load a random patient from the existing dataset as an example
                sample_pid = df["patient_id"].sample(1).iloc[0]
                sample_pt  = df[df["patient_id"] == sample_pid]\
                               .sort_values("timestamp_day")
                st.session_state.new_patient_draws = sample_pt.to_dict("records")
                st.rerun()

        # ── Live preview ──────────────────────────────────────────────────────
        draws = st.session_state.new_patient_draws

        if len(draws) == 0:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#4a5568;
                        border: 1px dashed rgba(255,255,255,0.1);
                        border-radius:12px; margin-top:1rem">
                <p style="font-size:2rem; margin:0">💉</p>
                <p style="font-family:'DM Mono',monospace; font-size:0.8rem;
                          letter-spacing:0.1em; text-transform:uppercase;
                          margin-top:0.5rem">
                    No draws yet — add a CBC draw above
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif len(draws) < 15:
            remaining = 15 - len(draws)
            st.info(
                f"📋 {len(draws)} draw{'s' if len(draws)>1 else ''} entered. "
                f"Add {remaining} more to enable anomaly scoring "
                f"(minimum window size = 15)."
            )
            # Show current draws table
            preview_df = pd.DataFrame(draws)
            st.dataframe(
                preview_df[["timestamp_day"] + CBC_FEATURES].round(2),
                use_container_width=True,
                height=220,
            )

        else:
            # Enough draws — run inference
            new_patient_df = pd.DataFrame(draws)\
                               .sort_values("timestamp_day")\
                               .reset_index(drop=True)

            st.markdown('<p class="section-label">Live anomaly scoring</p>',
                        unsafe_allow_html=True)

            with st.spinner("Running anomaly detection..."):
                new_pred_df = get_patient_predictions(
                    new_patient_df, model, preprocessor, device
                )

            # Peak score summary
            if new_pred_df is not None:
                max_s = new_pred_df["anomaly_score"].dropna().max()
                risk_c = ("risk-high"   if max_s >= 0.7 else
                          "risk-medium" if max_s >= 0.4 else "risk-low")
                risk_l = ("HIGH"   if max_s >= 0.7 else
                          "MEDIUM" if max_s >= 0.4 else "LOW")

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Draws entered</div>
                        <div class="metric-value">{len(new_patient_df)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Peak anomaly score</div>
                        <div class="metric-value {risk_c}">{max_s:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Risk level</div>
                        <div class="metric-value {risk_c}">{risk_l}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Feature dropdown for new patient
            new_feature = st.selectbox(
                "CBC variable to overlay",
                CBC_FEATURES,
                format_func=lambda f: f"{f}  ({UNITS[f]})",
                key="new_patient_feature",
            )

            fig_new = build_overlay_chart(
                new_patient_df, new_pred_df, new_feature
            )
            st.plotly_chart(fig_new, use_container_width=True)

            # All features overview for new patient
            fig_all_new = build_all_features_chart(new_patient_df, new_pred_df)
            st.markdown(
                '<p class="section-label">All features overview</p>',
                unsafe_allow_html=True
            )
            st.plotly_chart(fig_all_new, use_container_width=True)

            # Download predictions
            if new_pred_df is not None:
                merged_new = new_patient_df.merge(
                    new_pred_df[["timestamp_day", "anomaly_score", "risk_level"]],
                    on="timestamp_day", how="left"
                )
                csv = merged_new.to_csv(index=False)
                st.download_button(
                    "⬇️  Download predictions CSV",
                    data=csv,
                    file_name="new_patient_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()

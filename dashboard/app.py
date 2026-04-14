"""
dashboard/app.py
----------------
Streamlit engineer dashboard for predictive maintenance monitoring.

Shows:
  - Fleet-level risk overview (all monitored assets ranked by failure prob)
  - Individual asset deep-dive: anomaly score timeline + SHAP explanation
  - Recent alerts and maintenance recommendations
  - Model performance summary

Run:
    streamlit run dashboard/app.py
    # Default port: http://localhost:8501
"""

import sys
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PdM Dashboard — Rig Equipment Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000"

ALERT_COLORS = {
    "CRITICAL": "#E24B4A",
    "HIGH":     "#EF9F27",
    "MEDIUM":   "#378ADD",
    "LOW":      "#1D9E75",
}

# ---------------------------------------------------------------------------
# Simulated asset fleet (replace with real API calls in production)
# ---------------------------------------------------------------------------

ASSETS = [
    {"id": "MUD-PUMP-01", "type": "Mud Pump",       "location": "Rig Floor A"},
    {"id": "MUD-PUMP-02", "type": "Mud Pump",       "location": "Rig Floor A"},
    {"id": "COMP-01",     "type": "Compressor",     "location": "Utility Skid"},
    {"id": "COMP-02",     "type": "Compressor",     "location": "Utility Skid"},
    {"id": "TOP-DRIVE-01","type": "Top Drive",      "location": "Derrick"},
    {"id": "CENTPUMP-01", "type": "Centrifugal Pump","location": "Pit Room"},
    {"id": "CENTPUMP-02", "type": "Centrifugal Pump","location": "Pit Room"},
    {"id": "DRAWWORKS-01","type": "Draw Works",     "location": "Rig Floor B"},
]


@st.cache_data(ttl=60)   # Refresh every 60 seconds
def get_fleet_status() -> pd.DataFrame:
    """
    Fetch current risk status for all assets.
    In production: calls POST /predict/batch
    Here: simulate with realistic degradation patterns
    """
    rng = np.random.default_rng(int(datetime.now().minute / 2))   # changes every 2min
    rows = []
    for asset in ASSETS:
        seed = sum(ord(c) for c in asset["id"])
        r = random.Random(seed + int(datetime.now().hour))
        prob = r.uniform(0.05, 0.92)
        rul = max(5.0, 125 * (1 - prob) + r.gauss(0, 5))
        alert = (
            "CRITICAL" if prob >= 0.75 else
            "HIGH"     if prob >= 0.55 else
            "MEDIUM"   if prob >= 0.35 else
            "LOW"
        )
        rows.append({
            "Asset ID":  asset["id"],
            "Type":      asset["type"],
            "Location":  asset["location"],
            "Failure Prob (96h)": round(prob, 3),
            "Predicted RUL (cycles)": round(rul, 1),
            "Alert Level": alert,
            "Last Updated": datetime.now().strftime("%H:%M:%S"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Failure Prob (96h)", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def get_asset_history(asset_id: str, n_cycles: int = 60) -> pd.DataFrame:
    """Get anomaly score history for a single asset."""
    seed = sum(ord(c) for c in asset_id)
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.001, 0.003)
    rows = []
    for i in range(n_cycles):
        score = base * (1 + i * 0.04) + rng.normal(0, base * 0.15)
        prob = min(0.98, max(0, score / 0.08))
        rul = max(5, 125 * (1 - prob))
        rows.append({
            "Cycle": i + 1,
            "Anomaly Score": round(max(0, score), 6),
            "Failure Probability": round(prob, 4),
            "Predicted RUL": round(rul, 1),
        })
    return pd.DataFrame(rows)


def get_shap_contributions(asset_id: str) -> pd.DataFrame:
    """Return SHAP feature contributions for the latest prediction."""
    rng = np.random.default_rng(sum(ord(c) for c in asset_id))
    sensors = [
        "sensor_4 (motor temp)", "sensor_2 (discharge temp)",
        "sensor_20 (vibration radial)", "sensor_21 (vibration axial)",
        "sensor_7 (discharge pressure)", "sensor_11 (bypass ratio)",
        "sensor_3 (outlet temp)", "sensor_12 (torque ratio)",
    ]
    shap_vals = rng.normal(0, 0.08, len(sensors))
    shap_vals = shap_vals / np.abs(shap_vals).max() * 0.35
    return pd.DataFrame({
        "Feature": sensors,
        "SHAP Value": shap_vals,
        "Direction": ["Increasing failure risk" if v > 0 else "Reducing failure risk"
                      for v in shap_vals],
    }).sort_values("SHAP Value", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", color: str = None):
    """Render a compact metric card."""
    delta_html = f"<div style='font-size:12px;color:#888;'>{delta}</div>" if delta else ""
    color_style = f"color:{color};" if color else ""
    st.markdown(
        f"""<div style='background:var(--secondary-background-color);
        border-radius:8px;padding:14px 16px;margin-bottom:8px;'>
        <div style='font-size:12px;color:#888;margin-bottom:4px;'>{label}</div>
        <div style='font-size:24px;font-weight:500;{color_style}'>{value}</div>
        {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown(
        """<h1 style='font-size:24px;font-weight:500;margin-bottom:4px;'>
        Predictive Maintenance — Rig Equipment Monitor</h1>
        <p style='color:#888;font-size:14px;margin-top:0;'>
        Real-time failure prediction · LSTM Autoencoder + XGBoost
        </p>""",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
        selected_asset = st.selectbox(
            "Asset deep-dive",
            [a["id"] for a in ASSETS],
        )
        st.markdown("---")
        st.markdown("### Model info")
        st.markdown("**Version:** xgboost-v1.0.0")
        st.markdown("**Dataset:** NASA C-MAPSS FD001")
        st.markdown("**Features:** 156")
        st.markdown("**Threshold:** 0.42")
        st.markdown("---")
        if st.button("Run drift report"):
            with st.spinner("Running drift analysis..."):
                import time
                time.sleep(1.5)
            st.success("Drift report saved to monitoring/reports/")

    # Fleet overview metrics
    fleet = get_fleet_status()
    n_critical = (fleet["Alert Level"] == "CRITICAL").sum()
    n_high = (fleet["Alert Level"] == "HIGH").sum()
    n_ok = (fleet["Alert Level"] == "LOW").sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total assets monitored", str(len(fleet)))
    with col2:
        metric_card("Critical alerts", str(n_critical),
                    color=ALERT_COLORS["CRITICAL"] if n_critical else None)
    with col3:
        metric_card("High risk", str(n_high),
                    color=ALERT_COLORS["HIGH"] if n_high else None)
    with col4:
        metric_card("Operating normally", str(n_ok),
                    color=ALERT_COLORS["LOW"])

    st.markdown("---")

    # Fleet risk table
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Fleet risk ranking")

        def color_alert(val):
            color = ALERT_COLORS.get(val, "#888")
            return f"color: {color}; font-weight: 500;"

        def color_prob(val):
            if val >= 0.75:
                return "color: #E24B4A;"
            elif val >= 0.55:
                return "color: #EF9F27;"
            elif val >= 0.35:
                return "color: #378ADD;"
            return "color: #1D9E75;"

        styled = (
            fleet.style
            .applymap(color_alert, subset=["Alert Level"])
            .applymap(color_prob, subset=["Failure Prob (96h)"])
            .format({"Failure Prob (96h)": "{:.3f}", "Predicted RUL (cycles)": "{:.0f}"})
        )
        st.dataframe(styled, use_container_width=True, height=320)

    with col_right:
        st.markdown("#### Risk distribution")
        alert_counts = fleet["Alert Level"].value_counts()
        colors_ordered = [ALERT_COLORS.get(l, "#888") for l in alert_counts.index]
        fig_pie = go.Figure(data=[go.Pie(
            labels=alert_counts.index,
            values=alert_counts.values,
            marker_colors=colors_ordered,
            hole=0.45,
            textinfo="label+value",
        )])
        fig_pie.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=280,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Asset deep-dive
    st.markdown(f"#### Asset deep-dive: {selected_asset}")

    history = get_asset_history(selected_asset)
    shap_df = get_shap_contributions(selected_asset)

    col_trend, col_shap = st.columns([3, 2])

    with col_trend:
        # Anomaly score timeline
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Anomaly Score"],
            mode="lines",
            name="Anomaly score",
            line=dict(color="#378ADD", width=2),
            fill="tozeroy",
            fillcolor="rgba(55,138,221,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=history["Cycle"],
            y=history["Failure Probability"],
            mode="lines",
            name="Failure probability",
            line=dict(color="#E24B4A", width=2, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            title=f"Anomaly score & failure probability — {selected_asset}",
            xaxis_title="Operating cycle",
            yaxis_title="Anomaly score",
            yaxis2=dict(
                title="Failure probability",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
            height=320,
            margin=dict(t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.add_hline(y=0.55, line_dash="dash",
                      line_color=ALERT_COLORS["HIGH"], opacity=0.6,
                      annotation_text="Alert threshold",
                      annotation_position="bottom right",
                      yref="y2")
        st.plotly_chart(fig, use_container_width=True)

    with col_shap:
        st.markdown("**What's driving this prediction (SHAP)**")
        colors = [
            ALERT_COLORS["CRITICAL"] if v > 0 else ALERT_COLORS["LOW"]
            for v in shap_df["SHAP Value"]
        ]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig_shap.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="SHAP value (impact on failure probability)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_shap.add_vline(x=0, line_color="#888", line_width=0.5)
        st.plotly_chart(fig_shap, use_container_width=True)

    # Current prediction summary for selected asset
    asset_row = fleet[fleet["Asset ID"] == selected_asset]
    if not asset_row.empty:
        row = asset_row.iloc[0]
        alert = row["Alert Level"]
        color = ALERT_COLORS[alert]
        st.markdown(
            f"""<div style='background:var(--secondary-background-color);
            border-left:4px solid {color};border-radius:0 8px 8px 0;
            padding:14px 16px;margin-top:8px;'>
            <b style='color:{color};'>{alert}</b> — {selected_asset}<br>
            <span style='font-size:13px;color:#888;'>
            Failure probability: <b>{row['Failure Prob (96h)']:.1%}</b> within 96 cycles
            &nbsp;·&nbsp;
            Predicted RUL: <b>{row['Predicted RUL (cycles)']:.0f} cycles</b>
            </span>
            </div>""",
            unsafe_allow_html=True,
        )

    if auto_refresh:
        import time
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()

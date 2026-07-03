"""
streamlit_app.py — The Living Sentiment Engine Dashboard
─────────────────────────────────────────────────────────
Tabs:
  1. 📰 Live Sentiment Feed   — today's headlines + labels (color-coded)
  2. 📈 Trend Chart           — daily positive/negative/neutral ratio (Plotly)
  3. 🩺 Model Health          — champion F1, retrain date, drift status
  4. 🌊 Drift Report          — embedded Evidently HTML report

Usage:
    streamlit run src/dashboard/streamlit_app.py
"""

import json
import os

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Living Sentiment Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp { background: #060B18 !important; }

.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

#MainMenu, footer, .stDeployButton { visibility: hidden !important; }

[data-testid="stSidebar"] { background: #0A0F1E !important; border-right: 1px solid #1A2235 !important; }

.metric-card {
  background: linear-gradient(135deg, #0D1528 0%, #111827 100%);
  border: 1px solid #1E2D45;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid #1A2235 !important;
  gap: 8px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: #4B5563 !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  border-radius: 8px 8px 0 0 !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  color: #FFFFFF !important;
  background: linear-gradient(180deg, #1A2744 0%, transparent 100%) !important;
  border-bottom: 2px solid #3B82F6 !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div {
  background: #0D1528 !important;
  border: 1px solid #1E2D45 !important;
  border-radius: 10px !important;
  color: white !important;
}

.stButton > button {
  background: linear-gradient(135deg, #1D4ED8, #6366F1) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 10px 20px !important;
  width: 100% !important;
  box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
  transition: all 0.2s ease !important;
}

.stTextArea textarea {
  background: #0D1528 !important;
  border: 1px solid #1E2D45 !important;
  border-radius: 10px !important;
  color: white !important;
  font-size: 13px !important;
}

.stDataFrame, [data-testid="stDataFrame"] iframe,
.stDataFrame > div, .stDataFrame table {
  background: #0D1528 !important;
  color: white !important;
  border: 1px solid #1E2D45 !important;
  border-radius: 12px !important;
}

@keyframes pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px currentColor; }
  50% { opacity: 0.6; box-shadow: 0 0 16px currentColor; }
}
@keyframes pulse-orange {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.75; }
}
"""
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")

@st.cache_data(ttl=60)
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = load_config()
API_BASE = os.environ.get("API_BASE_URL", f"http://localhost:{cfg['serving']['port']}")

# ── Helpers ───────────────────────────────────────────────────────────────────
LABEL_COLORS = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}
LABEL_EMOJIS = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
LABEL_STYLES = {
    "positive": {"color": "#22C55E", "bg": "rgba(34,197,94,0.15)"},
    "negative": {"color": "#EF4444", "bg": "rgba(239,68,68,0.15)"},
    "neutral": {"color": "#94A3B8", "bg": "rgba(148,163,184,0.15)"},
}
CHART_COLORS = {"positive": "#10B981", "negative": "#F43F5E", "neutral": "#6366F1"}


def apply_plotly_theme(fig, height=420, legend_orientation="h", legend_y=-0.15):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,21,40,0.8)",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(gridcolor="#1A2235", linecolor="#1A2235", tickfont=dict(color="#4B5563")),
        yaxis=dict(gridcolor="#1A2235", linecolor="#1A2235", tickfont=dict(color="#4B5563")),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            orientation=legend_orientation,
            y=legend_y,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#0D1528", font_color="white", bordercolor="#1E2D45"),
        height=height,
    )
    return fig


@st.cache_data(ttl=300)
def load_labeled_data():
    labeled_path = Path(cfg["data"]["labeled_file"])
    if not labeled_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(labeled_path, dtype=str).dropna(subset=["title"])
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["date"] = df["publishedAt"].dt.date
    return df


@st.cache_data(ttl=60)
def load_model_metadata():
    champion_dir = Path(cfg["model"]["champion_dir"])
    meta_path = champion_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


@st.cache_data(ttl=60)
def load_latest_drift_summary():
    reports_dir = Path(cfg["monitoring"]["reports_dir"])
    summaries = sorted(reports_dir.glob("drift_summary_*.json"), reverse=True)
    if summaries:
        return json.loads(summaries[0].read_text())
    return {}


def api_predict(headline: str) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}/predict", json={"headline": headline}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def api_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


health = api_health()
model_meta = load_model_metadata()
model_version = (
    health.get("model_version", model_meta.get("version", "?"))
    if health
    else model_meta.get("version", "?")
)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0">
      <span style="width:8px;height:8px;background:#3B82F6;border-radius:50%;display:inline-block;
                   box-shadow:0 0 8px #3B82F6;animation:pulse 2s infinite;flex-shrink:0"></span>
      <p style="font-size:16px;font-weight:700;color:white;margin:0;letter-spacing:-0.3px">
        Living Sentiment Engine
      </p>
    </div>
    <p style="font-size:11px;color:#4B5563;margin:4px 0 0 16px">
      Real-time financial intelligence via FinBERT
    </p>
    <hr style="border:none;border-top:1px solid #1A2235;margin:16px 0">
    """, unsafe_allow_html=True)

    if health:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;background:#061A0E;border:1px solid #16A34A;
                    border-radius:20px;padding:8px 14px;width:fit-content;margin-bottom:12px">
          <span style="width:8px;height:8px;background:#22C55E;border-radius:50%;display:inline-block;
                       box-shadow:0 0 8px #22C55E;animation:pulse 2s infinite"></span>
          <span style="color:#4ADE80;font-size:13px;font-weight:600">API Online</span>
        </div>
        <div style="background:#0D1528;border:1px solid #1E2D45;border-radius:12px;padding:16px;margin-bottom:12px">
          <p style="color:#4B5563;font-size:10px;font-weight:600;letter-spacing:1px;margin:0;text-transform:uppercase">
            Model Version
          </p>
          <p style="color:white;font-size:15px;font-weight:700;margin:4px 0 0 0">
            v{health.get('model_version', '?')}
          </p>
          <p style="color:#4B5563;font-size:10px;font-weight:600;letter-spacing:1px;margin:12px 0 0 0;text-transform:uppercase">
            Uptime
          </p>
          <p style="color:white;font-size:15px;font-weight:700;margin:4px 0 0 0">
            {health.get('uptime_seconds', 0):.0f}s
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;background:#1A0A0A;border:1px solid #DC2626;
                    border-radius:20px;padding:8px 14px;width:fit-content;margin-bottom:12px">
          <span style="width:8px;height:8px;background:#EF4444;border-radius:50%;display:inline-block"></span>
          <span style="color:#F87171;font-size:13px;font-weight:600">API Offline</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0D1528;border:1px solid #1E2D45;border-radius:12px;padding:16px;margin-bottom:6px">
      <p style="color:#4B5563;font-size:11px;font-weight:600;letter-spacing:0.5px;margin:0 0 12px 0;text-transform:uppercase">
        Quick Predict
      </p>
    """, unsafe_allow_html=True)
    quick_input = st.text_area(
        "Enter headline:",
        placeholder="Fed raises interest rates by 25bps...",
        height=80,
        label_visibility="collapsed",
    )
    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if quick_input.strip():
            result = api_predict(quick_input.strip())
            if result:
                label = result["label"]
                style = LABEL_STYLES.get(label, LABEL_STYLES["neutral"])
                st.markdown(
                    f"<div style='background:{style['bg']};border-left:3px solid {style['color']};"
                    f"padding:10px 12px;border-radius:8px;margin-top:8px;'>"
                    f"<div style='font-weight:700;color:{style['color']};font-size:14px;'>"
                    f"{LABEL_EMOJIS[label]} {label.upper()}</div>"
                    f"<div style='color:#4B5563;font-size:12px;margin-top:4px;'>"
                    f"Confidence: {result['confidence']:.1%}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error("API unavailable")
        else:
            st.warning("Please enter a headline")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #1A2235;margin:16px 0'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
  <div>
    <h1 style="font-size:28px;font-weight:700;color:white;margin:0;letter-spacing:-0.5px">
      🧠 The Living Sentiment Engine
    </h1>
    <p style="color:#4B5563;font-size:13px;margin:4px 0 0 0">
      Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
  </div>
  <div style="background:#0D1528;border:1px solid #1E2D45;border-radius:10px;padding:8px 16px;text-align:right">
    <p style="color:#4B5563;font-size:11px;margin:0">MODEL</p>
    <p style="color:#6366F1;font-size:13px;font-weight:600;margin:0">v{model_version}</p>
  </div>
</div>
<hr style="border:none;border-top:1px solid #1A2235;margin:16px 0 24px 0">
""", unsafe_allow_html=True)

tab1, tab2, tab_ent, tab3, tab4 = st.tabs([
    "📰 Live Feed", "📈 Trends", "🏢 Entities", "🩺 Model Health", "🌊 Drift Report"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Live Sentiment Feed
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:600;color:#FFFFFF;"
        "border-left:3px solid #3B82F6;padding-left:12px;margin-bottom:1rem;'>Today's Headlines</h2>",
        unsafe_allow_html=True,
    )

    df = load_labeled_data()
    if df.empty:
        st.info("No data yet. Run `newsapi_scraper.py` and `label_pipeline.py` to populate.")
    else:
        date_options = sorted(df["date"].dropna().unique(), reverse=True)
        selected_date = st.selectbox("Select date:", date_options, index=0)
        day_df = df[df["date"] == selected_date].sort_values("publishedAt", ascending=False)

        total = len(day_df)
        pos = (day_df["label"] == "positive").sum()
        neg = (day_df["label"] == "negative").sum()
        neu = (day_df["label"] == "neutral").sum()

        col1, col2, col3, col4 = st.columns(4)
        _card_base = (
            "background:linear-gradient(135deg,#0D1528,#111827);"
            "border:1px solid #1E2D45;border-radius:16px;padding:20px 24px;height:110px"
        )
        with col1:
            st.markdown(
                f"<div style='{_card_base};border-top:3px solid #3B82F6'>"
                f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
                f"TOTAL HEADLINES</p>"
                f"<p style='color:white;font-size:36px;font-weight:700;margin:8px 0 0 0;line-height:1'>{total}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            pct = f"{pos / total:.0%} of total" if total else ""
            st.markdown(
                f"<div style='{_card_base};border-top:3px solid #22C55E'>"
                f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
                f"POSITIVE</p>"
                f"<p style='color:white;font-size:36px;font-weight:700;margin:8px 0 0 0;line-height:1'>{pos}</p>"
                f"<p style='color:#4B5563;font-size:12px;margin:4px 0 0 0'>{pct}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            pct = f"{neg / total:.0%} of total" if total else ""
            st.markdown(
                f"<div style='{_card_base};border-top:3px solid #EF4444'>"
                f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
                f"NEGATIVE</p>"
                f"<p style='color:white;font-size:36px;font-weight:700;margin:8px 0 0 0;line-height:1'>{neg}</p>"
                f"<p style='color:#4B5563;font-size:12px;margin:4px 0 0 0'>{pct}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col4:
            pct = f"{neu / total:.0%} of total" if total else ""
            st.markdown(
                f"<div style='{_card_base};border-top:3px solid #94A3B8'>"
                f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
                f"NEUTRAL</p>"
                f"<p style='color:white;font-size:36px;font-weight:700;margin:8px 0 0 0;line-height:1'>{neu}</p>"
                f"<p style='color:#4B5563;font-size:12px;margin:4px 0 0 0'>{pct}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        label_filter = st.multiselect(
            "Filter by label:", ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        filtered = day_df[day_df["label"].isin(label_filter)]

        for _, row in filtered.head(50).iterrows():
            label = row.get("label", "neutral")
            confidence = row.get("confidence", "")
            style = LABEL_STYLES.get(label, LABEL_STYLES["neutral"])
            conf_str = f"{float(confidence):.0%}" if confidence else "—"

            st.markdown(
                f"<div style='background:#0D1528;border:1px solid #1E2D45;border-left:3px solid {style['color']};"
                f"border-radius:12px;padding:16px 20px;margin-bottom:10px'>"
                f"<p style='color:white;font-size:14px;font-weight:500;margin:0;line-height:1.5'>"
                f"{row.get('title', '')}</p>"
                f"<div style='display:flex;gap:12px;margin-top:8px;align-items:center;flex-wrap:wrap'>"
                f"<span style='color:#4B5563;font-size:12px'>{row.get('source', '')}</span>"
                f"<span style='color:{style['color']};font-size:12px;font-weight:600;"
                f"background:{style['bg']};padding:2px 10px;border-radius:20px'>{label.upper()}</span>"
                f"<span style='color:#4B5563;font-size:12px'>{conf_str}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Trend Chart
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:600;color:#FFFFFF;"
        "border-left:3px solid #3B82F6;padding-left:12px;margin-bottom:1rem;'>Sentiment Trends Over Time</h2>",
        unsafe_allow_html=True,
    )

    df = load_labeled_data()
    if df.empty or "date" not in df.columns:
        st.info("No data yet.")
    else:
        daily = (
            df.groupby(["date", "label"])
            .size()
            .reset_index(name="count")
            .pivot(index="date", columns="label", values="count")
            .fillna(0)
            .reset_index()
        )

        for col in ["positive", "negative", "neutral"]:
            if col not in daily.columns:
                daily[col] = 0

        daily["total"] = daily[["positive", "negative", "neutral"]].sum(axis=1)
        for col in ["positive", "negative", "neutral"]:
            daily[f"{col}_pct"] = daily[col] / daily["total"].replace(0, 1)

        view = st.radio("View mode:", ["Absolute counts", "Percentage"], horizontal=True)

        fig = go.Figure()
        if view == "Absolute counts":
            for label, color in CHART_COLORS.items():
                fig.add_trace(go.Bar(
                    x=daily["date"], y=daily[label],
                    name=label.capitalize(), marker_color=color,
                ))
            fig.update_layout(barmode="stack", yaxis_title="Headlines")
        else:
            for label, color in CHART_COLORS.items():
                fig.add_trace(go.Scatter(
                    x=daily["date"], y=daily[f"{label}_pct"],
                    name=label.capitalize(), marker_color=color,
                    stackgroup="one", fill="tonexty",
                ))
            fig.update_layout(yaxis_title="Percentage", yaxis_tickformat=".0%")

        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Raw daily counts"):
            st.dataframe(daily.sort_values("date", ascending=False), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB ENT: Entity Sentiment
# ──────────────────────────────────────────────────────────────────────────────
with tab_ent:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:600;color:#FFFFFF;"
        "border-left:3px solid #3B82F6;padding-left:12px;margin-bottom:1rem;'>🏢 Entity Sentiment</h2>",
        unsafe_allow_html=True,
    )

    df = load_labeled_data()
    if df.empty or "entities" not in df.columns:
        st.info("No entity data available yet. Run the updated pipeline.")
    else:
        ent_rows = []
        for _, row in df.dropna(subset=["entities"]).iterrows():
            ents = [e.strip() for e in str(row["entities"]).split(",") if e.strip()]
            for e in ents:
                ent_rows.append({"entity": e, "label": row["label"]})

        if not ent_rows:
            st.info("No entities extracted yet.")
        else:
            ent_df = pd.DataFrame(ent_rows)
            top_ents = ent_df["entity"].value_counts().head(15).index.tolist()
            top_ent_df = ent_df[ent_df["entity"].isin(top_ents)]
            ent_summary = top_ent_df.groupby(["entity", "label"]).size().unstack(fill_value=0)

            for col in ["positive", "negative", "neutral"]:
                if col not in ent_summary.columns:
                    ent_summary[col] = 0

            ent_summary["total"] = ent_summary.sum(axis=1)
            ent_summary = ent_summary.sort_values("total", ascending=True)

            fig = go.Figure()
            for label, color in CHART_COLORS.items():
                fig.add_trace(go.Bar(
                    y=ent_summary.index,
                    x=ent_summary[label],
                    name=label.capitalize(),
                    marker_color=color,
                    orientation="h",
                ))
            fig.update_layout(
                barmode="stack",
                title=dict(text="Top 15 Most Discussed Entities", font=dict(color="#E2E8F0", size=14)),
            )
            apply_plotly_theme(fig, height=600, legend_y=-0.1)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Health
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:600;color:#FFFFFF;"
        "border-left:3px solid #3B82F6;padding-left:12px;margin-bottom:1rem;'>🩺 Model Health Dashboard</h2>",
        unsafe_allow_html=True,
    )

    meta = load_model_metadata()
    drift = load_latest_drift_summary()
    val_path = Path("reports/validation_report.json")
    val_data = json.loads(val_path.read_text()) if val_path.exists() else {}

    col1, col2, col3 = st.columns(3)
    _kpi_base = (
        "background:linear-gradient(135deg,#0D1528,#111827);"
        "border:1px solid #1E2D45;border-radius:16px;padding:20px 24px;"
        "box-shadow:0 0 20px rgba(59,130,246,0.1)"
    )
    _f1 = f"{meta.get('macro_f1', 'N/A'):.4f}" if meta.get("macro_f1") else "N/A"
    _drift_val = (
        f"{drift.get('drift_score', 'N/A'):.4f}"
        if drift.get("drift_score") is not None
        else "N/A"
    )
    _drift_col = "#EF4444" if drift.get("drift_detected") else "#22C55E"
    with col1:
        st.markdown(
            f"<div style='{_kpi_base};border-top:3px solid #3B82F6'>"
            f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
            f"CHAMPION MACRO F1</p>"
            f"<p style='color:#3B82F6;font-size:40px;font-weight:700;margin:8px 0 0 0;line-height:1'>{_f1}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='{_kpi_base};border-top:3px solid #8B5CF6'>"
            f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
            f"TRAINING SAMPLES</p>"
            f"<p style='color:#A78BFA;font-size:40px;font-weight:700;margin:8px 0 0 0;line-height:1'>"
            f"{meta.get('train_samples', 'N/A')}</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div style='{_kpi_base};border-top:3px solid {_drift_col}'>"
            f"<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:1px;margin:0'>"
            f"DRIFT SCORE</p>"
            f"<p style='color:{_drift_col};font-size:40px;font-weight:700;margin:8px 0 0 0;line-height:1'>"
            f"{_drift_val}</p></div>",
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div style='background:#0D1528;border:1px solid #1E2D45;border-radius:12px;padding:18px'>"
            "<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:0.5px;"
            "margin:0 0 14px 0;text-transform:uppercase'>Validation Gate Results</p>",
            unsafe_allow_html=True,
        )
        if val_data:
            _op = val_data.get("overall_passed")
            _oc = "#22C55E" if _op else "#EF4444"
            _ot = "PASSED" if _op else "FAILED"
            st.markdown(
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"background:#111827;border:1px solid #1E2D45;border-radius:10px;padding:12px 16px;margin-bottom:12px'>"
                f"<span style='color:#E2E8F0;font-size:14px;font-weight:600'>Overall Gate</span>"
                f"<span style='color:{_oc};font-size:12px;font-weight:700;background:{_oc}22;"
                f"padding:4px 12px;border-radius:20px'>{_ot}</span></div>",
                unsafe_allow_html=True,
            )
            for check_name, check_data in val_data.get("checks", {}).items():
                _p = check_data.get("passed")
                _rc = "#22C55E" if _p else "#EF4444"
                _status = "PASS" if _p else "FAIL"
                _icon = "✓" if _p else "✗"
                display_name = check_name.replace("_", " ").title()

                # Extract extra metrics if present
                _metrics = ""
                if check_name == "champion_comparison":
                    cand = check_data.get('candidate_f1', 0)
                    champ = check_data.get('champion_f1', 0)
                    _metrics = f"<div style='color:#64748B;font-size:11px;margin-top:2px;'>Candidate: {cand:.4f} vs Champion: {champ:.4f}</div>"
                elif check_name == "absolute_f1":
                    cand = check_data.get('candidate_f1', 0)
                    thresh = check_data.get('threshold', 0)
                    _metrics = f"<div style='color:#64748B;font-size:11px;margin-top:2px;'>Score: {cand:.4f} (Threshold: {thresh:.2f})</div>"

                st.markdown(
                    f"<div style='display:flex;align-items:center;justify-content:space-between;"
                    f"padding:10px 14px;margin-bottom:6px;background:#060B18;border:1px solid #1A2235;"
                    f"border-radius:10px'>"
                    f"<div style='display:flex;align-items:flex-start;gap:10px'>"
                    f"<span style='color:{_rc};font-size:16px;font-weight:700;width:20px;margin-top:-2px;'>{_icon}</span>"
                    f"<div>"
                    f"<div style='color:#E2E8F0;font-size:13px;font-weight:500'>{display_name}</div>"
                    f"{_metrics}"
                    f"</div>"
                    f"</div>"
                    f"<span style='color:{_rc};font-size:11px;font-weight:700;background:{_rc}22;"
                    f"padding:3px 10px;border-radius:20px'>{_status}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='color:#4B5563;font-size:13px;font-style:italic;margin:0'>"
                "No validation report found. Run <code>validate.py</code>.</p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(
            "<div style='background:#0D1528;border:1px solid #1E2D45;border-radius:12px;padding:18px'>"
            "<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:0.5px;"
            "margin:0 0 14px 0;text-transform:uppercase'>Drift Summary</p>",
            unsafe_allow_html=True,
        )
        if drift:
            drift_detected = drift.get("drift_detected", False)
            if drift_detected:
                st.markdown(
                    "<div style='background:linear-gradient(135deg,#1A0A00,#2D1500);"
                    "border:1px solid #F59E0B;border-radius:12px;padding:14px 20px;margin-bottom:14px;"
                    "animation:pulse-orange 2s infinite'>"
                    "<span style='color:#F59E0B;font-weight:700;font-size:14px'>"
                    "⚠ Data drift detected — review the Drift Report tab</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:rgba(34,197,94,0.1);border:1px solid #22C55E;"
                    "border-radius:10px;padding:12px 16px;margin-bottom:14px'>"
                    "<span style='color:#22C55E;font-weight:700;font-size:14px'>✓ No drift detected</span></div>",
                    unsafe_allow_html=True,
                )
            for _label_txt, _val_txt in [
                ("Score", str(drift.get("drift_score", "N/A"))),
                ("Threshold", str(drift.get("drift_threshold", "N/A"))),
                ("Date", str(drift.get("date", "N/A"))),
                ("Ref Rows", str(drift.get("reference_rows", "N/A"))),
                ("Cur Rows", str(drift.get("current_rows", "N/A"))),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
                    f"border-bottom:1px solid #1A2235'>"
                    f"<span style='color:#4B5563;font-size:12px'>{_label_txt}</span>"
                    f"<span style='color:#E2E8F0;font-size:12px;font-weight:600'>{_val_txt}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<p style='color:#4B5563;font-size:11px;font-weight:600;letter-spacing:0.5px;"
                "margin:14px 0 8px 0;text-transform:uppercase'>Current Label Distribution</p>",
                unsafe_allow_html=True,
            )
            cur_dist = drift.get("current_label_distribution", {})
            for label, pct in cur_dist.items():
                style = LABEL_STYLES.get(label, {"color": "#888", "bg": "transparent"})
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
                    f"border-bottom:1px solid #1A2235'>"
                    f"<span style='color:{style['color']};font-size:13px'>{label.capitalize()}</span>"
                    f"<span style='color:#E2E8F0;font-weight:600;font-size:13px'>{pct:.1%}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='color:#4B5563;font-size:13px;font-style:italic;margin:0'>"
                "No drift summary found. Run <code>drift_monitor.py</code>.</p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    mlflow_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    st.markdown(
        f"<p style='color:#4B5563;font-size:13px;margin:0'>"
        f"🔬 <b style='color:#94A3B8'>MLflow Experiments:</b> "
        f"<a href='{mlflow_uri}' target='_blank' style='color:#6366F1;text-decoration:none'>Open UI ↗</a></p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Drift Report
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:600;color:#FFFFFF;"
        "border-left:3px solid #3B82F6;padding-left:12px;margin-bottom:1rem;'>🌊 Evidently Data Drift Report</h2>",
        unsafe_allow_html=True,
    )

    reports_dir = Path(cfg["monitoring"]["reports_dir"])
    html_reports = sorted(reports_dir.glob("drift_report_*.html"), reverse=True)

    if not html_reports:
        st.info("No drift reports yet. Run `python src/monitoring/drift_monitor.py`.")
    else:
        report_options = {r.name: r for r in html_reports}
        selected_report = st.selectbox("Select report:", list(report_options.keys()))
        html_content = report_options[selected_report].read_text(encoding="utf-8")
        dark_override = """
<style>
  body, html { background: #0D1528 !important; color: #E2E8F0 !important; }
  table { background: #0D1528 !important; border-collapse: collapse; width:100%; }
  th { background: #1E2D45 !important; color: #94A3B8 !important; font-size:12px; padding:12px 16px; text-align:left; }
  td { background: #0D1528 !important; color: #E2E8F0 !important; padding:12px 16px; border-bottom: 1px solid #1A2235; }
  tr:hover td { background: #111827 !important; }
  h1,h2,h3,h4,p,span { color: #E2E8F0 !important; }
</style>
"""
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", dark_override + "</head>")
        else:
            html_content = dark_override + html_content
        components.html(html_content, height=500, scrolling=True)

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
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Living Sentiment Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Living Sentiment Engine")
    st.markdown("*Real-time financial news sentiment powered by FinBERT*")
    st.divider()

    health = api_health()
    if health:
        st.success("API Online ✅")
        st.caption(f"Model v{health.get('model_version', '?')}")
        st.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
    else:
        st.warning("API Offline ⚠️")

    st.divider()
    st.markdown("**Quick Predict**")
    quick_input = st.text_area("Enter headline:", placeholder="Fed raises interest rates by 25bps...", height=80)
    if st.button("🔍 Analyze", use_container_width=True):
        if quick_input.strip():
            result = api_predict(quick_input.strip())
            if result:
                label = result["label"]
                color = LABEL_COLORS[label]
                st.markdown(
                    f"<div style='background:{color}22; border-left:4px solid {color}; padding:8px; border-radius:4px;'>"
                    f"<b>{LABEL_EMOJIS[label]} {label.upper()}</b><br>"
                    f"Confidence: {result['confidence']:.1%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error("API unavailable")
        else:
            st.warning("Please enter a headline")

    st.divider()
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 The Living Sentiment Engine")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tab1, tab2, tab_ent, tab3, tab4 = st.tabs([
    "📰 Live Feed", "📈 Trends", "🏢 Entities", "🩺 Model Health", "🌊 Drift Report"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Live Sentiment Feed
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Today's Headlines")

    df = load_labeled_data()
    if df.empty:
        st.info("No data yet. Run `newsapi_scraper.py` and `label_pipeline.py` to populate.")
    else:
        # Filter to latest available date
        latest_date = df["date"].dropna().max()
        date_options = sorted(df["date"].dropna().unique(), reverse=True)
        selected_date = st.selectbox("Select date:", date_options, index=0)
        day_df = df[df["date"] == selected_date].sort_values("publishedAt", ascending=False)

        # Sentiment summary
        col1, col2, col3, col4 = st.columns(4)
        total = len(day_df)
        pos = (day_df["label"] == "positive").sum()
        neg = (day_df["label"] == "negative").sum()
        neu = (day_df["label"] == "neutral").sum()

        col1.metric("Total Headlines", total)
        col2.metric("🟢 Positive", f"{pos} ({pos/total:.0%})" if total else "0")
        col3.metric("🔴 Negative", f"{neg} ({neg/total:.0%})" if total else "0")
        col4.metric("⚪ Neutral", f"{neu} ({neu/total:.0%})" if total else "0")

        st.divider()

        # Label filter
        label_filter = st.multiselect(
            "Filter by label:", ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"]
        )
        filtered = day_df[day_df["label"].isin(label_filter)]

        for _, row in filtered.head(50).iterrows():
            label = row.get("label", "neutral")
            confidence = row.get("confidence", "")
            color = LABEL_COLORS.get(label, "#94a3b8")
            emoji = LABEL_EMOJIS.get(label, "⚪")
            conf_str = f"{float(confidence):.0%}" if confidence else ""

            ents_str = ""
            if "entities" in row and pd.notna(row["entities"]) and str(row["entities"]).strip():
                ents_str = f" | 🏢 {row['entities']}"

            st.markdown(
                f"""<div style='
                    background: {color}11;
                    border-left: 3px solid {color};
                    padding: 10px 14px;
                    border-radius: 6px;
                    margin-bottom: 6px;
                '>
                <span style='font-weight:600'>{emoji} {row.get('title','')}</span>
                <br>
                <small style='color:#888'>{row.get('source','')} — {label.upper()} {conf_str}{ents_str}</small>
                </div>""",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Trend Chart
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Sentiment Trends Over Time")

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
            for label, color in LABEL_COLORS.items():
                fig.add_trace(go.Bar(
                    x=daily["date"], y=daily[label],
                    name=label.capitalize(), marker_color=color,
                ))
            fig.update_layout(barmode="stack", yaxis_title="Headlines")
        else:
            for label, color in LABEL_COLORS.items():
                fig.add_trace(go.Scatter(
                    x=daily["date"], y=daily[f"{label}_pct"],
                    name=label.capitalize(), marker_color=color, stackgroup="one",
                ))
            fig.update_layout(yaxis_title="Percentage", yaxis_tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("📊 Raw daily counts"):
            st.dataframe(daily.sort_values("date", ascending=False), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB ENT: Entity Sentiment
# ──────────────────────────────────────────────────────────────────────────────
with tab_ent:
    st.subheader("🏢 Entity Sentiment")
    
    df = load_labeled_data()
    if df.empty or "entities" not in df.columns:
        st.info("No entity data available yet. Run the updated pipeline.")
    else:
        # Explode entities
        ent_rows = []
        for _, row in df.dropna(subset=["entities"]).iterrows():
            ents = [e.strip() for e in str(row["entities"]).split(",") if e.strip()]
            for e in ents:
                ent_rows.append({"entity": e, "label": row["label"]})
                
        if not ent_rows:
            st.info("No entities extracted yet.")
        else:
            ent_df = pd.DataFrame(ent_rows)
            
            # Count mentions
            top_ents = ent_df["entity"].value_counts().head(15).index.tolist()
            top_ent_df = ent_df[ent_df["entity"].isin(top_ents)]
            
            ent_summary = top_ent_df.groupby(["entity", "label"]).size().unstack(fill_value=0)
            
            for col in ["positive", "negative", "neutral"]:
                if col not in ent_summary.columns:
                    ent_summary[col] = 0
            
            ent_summary["total"] = ent_summary.sum(axis=1)
            ent_summary = ent_summary.sort_values("total", ascending=True)
            
            fig = go.Figure()
            for label, color in LABEL_COLORS.items():
                fig.add_trace(go.Bar(
                    y=ent_summary.index,
                    x=ent_summary[label],
                    name=label.capitalize(),
                    marker_color=color,
                    orientation="h"
                ))
            fig.update_layout(barmode="stack", title="Top 15 Most Discussed Entities", height=600)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Health
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🩺 Model Health Dashboard")

    meta = load_model_metadata()
    drift = load_latest_drift_summary()
    val_path = Path("reports/validation_report.json")
    val_data = json.loads(val_path.read_text()) if val_path.exists() else {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Champion Macro F1", f"{meta.get('macro_f1', 'N/A'):.4f}" if meta.get('macro_f1') else "N/A")
    col2.metric("Training Samples", meta.get("train_samples", "N/A"))
    col3.metric("Drift Score", f"{drift.get('drift_score', 'N/A'):.4f}" if drift.get('drift_score') is not None else "N/A")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Validation Gate Results**")
        if val_data:
            status = "✅ PASSED" if val_data.get("overall_passed") else "❌ FAILED"
            st.markdown(f"Overall: **{status}**")
            for check_name, check_data in val_data.get("checks", {}).items():
                icon = "✅" if check_data.get("passed") else "❌"
                st.markdown(f"- {icon} `{check_name}`")
        else:
            st.info("No validation report found. Run `validate.py`.")

    with col_b:
        st.markdown("**Drift Summary**")
        if drift:
            drift_detected = drift.get("drift_detected", False)
            st.markdown(f"Status: {'⚠️ Drift Detected' if drift_detected else '✅ No Drift'}")
            st.markdown(f"- Score: `{drift.get('drift_score', 'N/A')}`")
            st.markdown(f"- Threshold: `{drift.get('drift_threshold', 'N/A')}`")
            st.markdown(f"- Date: `{drift.get('date', 'N/A')}`")
            st.markdown("**Current label distribution:**")
            cur_dist = drift.get("current_label_distribution", {})
            for label, pct in cur_dist.items():
                color = LABEL_COLORS.get(label, "#888")
                st.markdown(
                    f"<span style='color:{color}'>{LABEL_EMOJIS.get(label,'')} {label}</span>: **{pct:.1%}**",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No drift summary found. Run `drift_monitor.py`.")

    # MLflow link
    st.divider()
    mlflow_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    st.markdown(f"🔬 **MLflow Experiments:** [Open UI]({mlflow_uri})")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Drift Report
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🌊 Evidently Data Drift Report")

    reports_dir = Path(cfg["monitoring"]["reports_dir"])
    html_reports = sorted(reports_dir.glob("drift_report_*.html"), reverse=True)

    if not html_reports:
        st.info("No drift reports yet. Run `python src/monitoring/drift_monitor.py`.")
    else:
        report_options = {r.name: r for r in html_reports}
        selected_report = st.selectbox("Select report:", list(report_options.keys()))
        html_content = report_options[selected_report].read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=800, scrolling=True)

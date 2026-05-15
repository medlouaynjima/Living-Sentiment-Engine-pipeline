"""
drift_monitor.py
────────────────
Uses Evidently AI (v0.7+) to detect data drift between:
  - Reference dataset: first chronological half of labeled data
  - Current dataset:   second chronological half (or last N days)

Outputs:
  - reports/drift/drift_report_YYYY-MM-DD.html
  - reports/drift/drift_summary_YYYY-MM-DD.json

Exits with code 1 (sets drift flag) if drift score > threshold.

Usage:
    python src/monitoring/drift_monitor.py
    python src/monitoring/drift_monitor.py --config configs/config.yaml
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Evidently 0.7+ imports (preset-based API) ────────────────────────────────
_EVIDENTLY_OK = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    _EVIDENTLY_OK = True
    log.info("Evidently loaded (preset API)")
except ImportError:
    log.warning("Evidently preset API not available — will use scipy fallback")

# ── Scipy fallback ────────────────────────────────────────────────────────────
try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_labeled_data(labeled_path: Path) -> pd.DataFrame:
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")
    df = pd.read_csv(labeled_path, dtype=str).dropna(subset=["title"])
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.sort_values("publishedAt").reset_index(drop=True)
    return df


def split_reference_current(df: pd.DataFrame, reference_days: int, current_days: int):
    now = df["publishedAt"].max()
    if pd.isna(now):
        n = len(df)
        split = max(1, n // 2)
        return df.iloc[:split].copy(), df.iloc[split:].copy()

    current_start = now - timedelta(days=current_days)
    reference_end = current_start
    reference_start = reference_end - timedelta(days=reference_days)

    reference = df[(df["publishedAt"] >= reference_start) & (df["publishedAt"] < reference_end)]
    current = df[df["publishedAt"] >= current_start]

    if len(reference) < 10 or len(current) < 10:
        log.warning(
            "Not enough data for time-based split (ref=%d, cur=%d). Using chronological halves.",
            len(reference), len(current),
        )
        n = len(df)
        split = max(1, n // 2)
        reference = df.iloc[:split]
        current = df.iloc[split:]

    return reference.copy(), current.copy()


def _add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_length"] = df["title"].str.len().fillna(0).astype(float)
    df["word_count"] = df["title"].str.split().str.len().fillna(0).astype(float)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    df["label_numeric"] = df["label"].map(label_map).fillna(0.0)
    return df


def compute_label_distribution(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts(normalize=True).to_dict()
    return {k: round(v, 4) for k, v in counts.items()}


# ── Evidently 0.7 report ──────────────────────────────────────────────────────
def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, report_path: Path) -> float:
    """Run Evidently DataDriftPreset report. Returns drift share (0-1)."""
    ref_feat = _add_numeric_features(reference)[["title_length", "word_count", "label_numeric"]]
    cur_feat = _add_numeric_features(current)[["title_length", "word_count", "label_numeric"]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_feat, current_data=cur_feat)
    report.save_html(str(report_path))
    log.info("Evidently drift report saved → %s", report_path)

    # Extract drift score from the result dict
    drift_score = 0.0
    try:
        result_dict = report.as_dict()
        for metric in result_dict.get("metrics", []):
            metric_id = str(metric.get("metric") or metric.get("metric_id") or "")
            if "DatasetDriftMetric" in metric_id or "DataDrift" in metric_id:
                res = metric.get("result", {})
                score = (
                    res.get("share_of_drifted_columns")
                    or res.get("drift_share")
                    or res.get("share_of_drifted_features")
                )
                if score is not None:
                    drift_score = float(score)
                    break
    except Exception as exc:
        log.warning("Could not extract drift score from Evidently result: %s", exc)

    return drift_score


# ── Scipy fallback report ─────────────────────────────────────────────────────
def run_scipy_report(reference: pd.DataFrame, current: pd.DataFrame, report_path: Path) -> float:
    """Simple KS-test drift detection. Generates a plain HTML summary."""
    ref_feat = _add_numeric_features(reference)
    cur_feat = _add_numeric_features(current)

    features = ["title_length", "word_count", "label_numeric"]
    results = []
    drifted = 0

    for feat in features:
        ks_stat, p_value = _scipy_stats.ks_2samp(ref_feat[feat], cur_feat[feat])
        is_drifted = p_value < 0.05
        if is_drifted:
            drifted += 1
        results.append({
            "feature": feat,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": is_drifted,
        })
        log.info("  %s — KS=%.4f  p=%.4f  %s", feat, ks_stat, p_value,
                 "⚠️ DRIFT" if is_drifted else "✅ OK")

    drift_share = drifted / len(features)

    # Write a minimal HTML report
    rows = "".join(
        f"<tr><td>{r['feature']}</td><td>{r['ks_statistic']}</td>"
        f"<td>{r['p_value']}</td><td>{'⚠️ YES' if r['drifted'] else '✅ NO'}</td></tr>"
        for r in results
    )
    html = f"""<!DOCTYPE html><html><head><title>Drift Report</title>
<style>body{{font-family:sans-serif;padding:2rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f4f4f4}}</style></head>
<body><h1>📊 Drift Monitor Report (scipy fallback)</h1>
<p>Reference rows: {len(reference)} | Current rows: {len(current)}</p>
<p>Drift share: <strong>{drift_share:.2%}</strong></p>
<table><tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drifted?</th></tr>
{rows}</table></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    log.info("Scipy drift report saved → %s", report_path)
    return drift_share


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run drift monitoring.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mon_cfg = cfg["monitoring"]
    labeled_path = Path(cfg["data"]["labeled_file"])
    reports_dir = Path(mon_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    report_html_path = reports_dir / f"drift_report_{today}.html"
    summary_json_path = reports_dir / f"drift_summary_{today}.json"

    # ── Load and split ────────────────────────────────────────────────────────
    df = load_labeled_data(labeled_path)
    log.info("Total labeled rows: %d", len(df))

    reference, current = split_reference_current(
        df,
        reference_days=mon_cfg["reference_days"],
        current_days=mon_cfg["current_days"],
    )
    log.info("Reference: %d rows  |  Current: %d rows", len(reference), len(current))

    # ── Run report ────────────────────────────────────────────────────────────
    if _EVIDENTLY_OK:
        try:
            drift_score = run_evidently_report(reference, current, report_html_path)
        except Exception as exc:
            log.warning("Evidently report failed (%s) — falling back to scipy.", exc)
            drift_score = run_scipy_report(reference, current, report_html_path) if _SCIPY_OK else 0.0
    elif _SCIPY_OK:
        drift_score = run_scipy_report(reference, current, report_html_path)
    else:
        log.error("Neither Evidently nor scipy available. Install scipy: pip install scipy")
        sys.exit(2)

    drift_threshold = mon_cfg["drift_threshold"]
    drift_detected = drift_score > drift_threshold

    # ── Label distribution change ─────────────────────────────────────────────
    ref_dist = compute_label_distribution(reference)
    cur_dist = compute_label_distribution(current)

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary = {
        "date": today,
        "drift_score": drift_score,
        "drift_threshold": drift_threshold,
        "drift_detected": drift_detected,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_label_distribution": ref_dist,
        "current_label_distribution": cur_dist,
        "report_html": str(report_html_path),
        "backend": "evidently" if _EVIDENTLY_OK else "scipy",
    }
    summary_json_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary saved → %s", summary_json_path)
    log.info(
        "Drift score: %.4f (threshold: %.2f) — %s",
        drift_score, drift_threshold,
        "⚠️  DRIFT DETECTED" if drift_detected else "✅ No drift",
    )

    if drift_detected:
        log.warning("Drift detected! Consider triggering retraining.")
        sys.exit(1)


if __name__ == "__main__":
    main()

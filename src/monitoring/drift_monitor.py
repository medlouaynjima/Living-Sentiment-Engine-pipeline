"""
drift_monitor.py
────────────────
Uses Evidently AI to detect data drift between:
  - Reference dataset: headlines from the first week of data
  - Current dataset: headlines from the last 7 days

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
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import TextOverviewPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
)
from evidently.report import Report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_labeled_data(labeled_path: Path) -> pd.DataFrame:
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")
    df = pd.read_csv(labeled_path, dtype=str).dropna(subset=["title"])
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.sort_values("publishedAt")
    return df


def split_reference_current(df: pd.DataFrame, reference_days: int, current_days: int):
    now = df["publishedAt"].max()
    if pd.isna(now):
        # Fall back to chronological split if dates are unavailable
        n = len(df)
        split = max(1, n // 2)
        return df.iloc[:split], df.iloc[split:]

    current_start = now - timedelta(days=current_days)
    reference_end = current_start
    reference_start = reference_end - timedelta(days=reference_days)

    reference = df[(df["publishedAt"] >= reference_start) & (df["publishedAt"] < reference_end)]
    current = df[df["publishedAt"] >= current_start]

    # Fallback: if either split is empty, use chronological halves
    if len(reference) < 10 or len(current) < 10:
        log.warning(
            "Not enough data for time-based split (ref=%d, cur=%d). Using chronological halves.",
            len(reference), len(current),
        )
        n = len(df)
        split = max(1, n // 2)
        reference = df.iloc[:split]
        current = df.iloc[split:]

    return reference, current


def compute_label_distribution(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts(normalize=True).to_dict()
    return {k: round(v, 4) for k, v in counts.items()}


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, report_path: Path):
    """Run Evidently text + drift report."""
    # Evidently works better with numeric features; use label encoding + title length
    for frame in [reference, current]:
        frame["title_length"] = frame["title"].str.len().fillna(0).astype(float)
        frame["word_count"] = frame["title"].str.split().str.len().fillna(0).astype(float)
        label_map = {"positive": 1, "negative": -1, "neutral": 0}
        frame["label_numeric"] = frame["label"].map(label_map).fillna(0).astype(float)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesSummaryMetric(),
        ColumnDriftMetric(column_name="title_length"),
        ColumnDriftMetric(column_name="word_count"),
        ColumnDriftMetric(column_name="label_numeric"),
    ])

    column_mapping = ColumnMapping(
        target="label_numeric",
        numerical_features=["title_length", "word_count", "label_numeric"],
    )

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    report.save_html(str(report_path))
    log.info("Drift report saved → %s", report_path)
    return report


def extract_drift_score(report: Report) -> float:
    """Extract the overall dataset drift share from the Evidently report."""
    try:
        result = report.as_dict()
        for metric in result.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                return float(metric["result"].get("share_of_drifted_columns", 0.0))
    except Exception as exc:
        log.warning("Could not extract drift score: %s", exc)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Evidently drift monitoring.")
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

    # ── Load and split data ───────────────────────────────────────────────────
    df = load_labeled_data(labeled_path)
    log.info("Total labeled rows: %d", len(df))

    reference, current = split_reference_current(
        df,
        reference_days=mon_cfg["reference_days"],
        current_days=mon_cfg["current_days"],
    )
    log.info("Reference: %d rows  |  Current: %d rows", len(reference), len(current))

    # ── Run Evidently ─────────────────────────────────────────────────────────
    report = run_evidently_report(reference, current, report_html_path)
    drift_score = extract_drift_score(report)
    drift_threshold = mon_cfg["drift_threshold"]
    drift_detected = drift_score > drift_threshold

    # ── Label distribution change ─────────────────────────────────────────────
    ref_dist = compute_label_distribution(reference)
    cur_dist = compute_label_distribution(current)

    # ── Write summary ─────────────────────────────────────────────────────────
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

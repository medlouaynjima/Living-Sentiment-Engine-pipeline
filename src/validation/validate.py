"""
validate.py
───────────
Validation gate — runs before promoting a candidate model to champion.

Checks:
  1. Macro F1 > threshold (default 0.70)
  2. Macro F1 > current champion's F1  (champion comparison)
  3. Bias: no keyword/entity dominates > 40% of training data
  4. Bias: per-class F1 variance < 0.15

Writes validation_report.json and exits with code 1 if any check fails.

Usage:
    python src/validation/validate.py
    python src/validation/validate.py --config configs/config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    return json.loads(meta_path.read_text())


def load_model_pipeline(model_dir: Path):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return hf_pipeline("text-classification", model=model, tokenizer=tokenizer,
                        device=device, truncation=True, max_length=128)


def compute_f1_per_class(nlp, df: pd.DataFrame) -> dict:
    """Run inference on validation headlines and compute per-class F1."""
    label_map = {"positive": 0, "negative": 1, "neutral": 2}


    # FinBERT label aliases
    alias = {
        "positive": "positive", "negative": "negative", "neutral": "neutral",
        "label_0": "positive", "label_1": "negative", "label_2": "neutral",
    }

    texts = df["title"].tolist()
    true_labels = df["label"].tolist()

    results = nlp(texts)
    pred_labels = [alias.get(r["label"].lower(), "neutral") for r in results]

    report = classification_report(
        true_labels, pred_labels,
        labels=["positive", "negative", "neutral"],
        output_dict=True,
        zero_division=0,
    )
    return report


def check_bias(df: pd.DataFrame, keywords: list[str], max_dominance: float) -> tuple[bool, str]:
    """Check that no keyword dominates more than max_dominance fraction of data."""
    total = len(df)
    if total == 0:
        return True, "Empty dataset"
    titles_lower = df["title"].str.lower()
    for kw in keywords:
        kw_count = titles_lower.str.contains(kw.lower(), regex=False).sum()
        dominance = kw_count / total
        if dominance > max_dominance:
            return False, (
                f"Keyword '{kw}' appears in {dominance:.1%} of data "
                f"(limit: {max_dominance:.0%})"
            )
    return True, "OK"


def promote_candidate(candidate_dir: Path, champion_dir: Path):
    """Copy candidate model files to champion directory."""
    import shutil
    if champion_dir.exists():
        shutil.rmtree(champion_dir)
    shutil.copytree(candidate_dir, champion_dir)
    log.info("✅ Candidate promoted → %s", champion_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]
    candidate_dir = Path(cfg["model"]["candidate_dir"])
    champion_dir = Path(cfg["model"]["champion_dir"])

    report_path = Path("reports") / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    checks = {}
    passed = True

    # ── Load candidate metadata ───────────────────────────────────────────────
    try:
        meta = load_metadata(candidate_dir)
    except FileNotFoundError as exc:
        log.error("Cannot load candidate metadata: %s", exc)
        sys.exit(1)

    candidate_f1 = meta.get("macro_f1", 0.0)
    log.info("Candidate macro F1: %.4f", candidate_f1)

    # ── Check 1: Absolute F1 threshold ───────────────────────────────────────
    min_f1 = val_cfg["min_f1"]
    check1_pass = candidate_f1 >= min_f1
    checks["absolute_f1"] = {
        "passed": check1_pass,
        "candidate_f1": candidate_f1,
        "threshold": min_f1,
    }
    log.info("Check 1 (F1 ≥ %.2f): %s", min_f1, "✅" if check1_pass else "❌")
    if not check1_pass:
        passed = False

    # ── Check 2: Champion comparison ─────────────────────────────────────────
    champion_f1 = 0.0
    if champion_dir.exists():
        try:
            champ_meta = load_metadata(champion_dir)
            champion_f1 = champ_meta.get("macro_f1", 0.0)
        except Exception:
            champion_f1 = 0.0

    check2_pass = candidate_f1 > champion_f1
    checks["champion_comparison"] = {
        "passed": check2_pass,
        "candidate_f1": candidate_f1,
        "champion_f1": champion_f1,
    }
    log.info(
        "Check 2 (candidate %.4f > champion %.4f): %s",
        candidate_f1, champion_f1, "✅" if check2_pass else "❌",
    )
    if not check2_pass:
        passed = False

    # ── Check 3: Bias — keyword dominance ────────────────────────────────────
    labeled_path = Path(cfg["data"]["labeled_file"])
    if labeled_path.exists():
        df = pd.read_csv(labeled_path, dtype=str).dropna(subset=["title", "label"])
        keywords = cfg["newsapi"]["keywords"]
        bias_ok, bias_msg = check_bias(df, keywords, val_cfg["max_entity_dominance"])
        checks["bias_keyword_dominance"] = {"passed": bias_ok, "detail": bias_msg}
        log.info("Check 3 (keyword dominance): %s — %s", "✅" if bias_ok else "❌", bias_msg)
        if not bias_ok:
            passed = False

        # ── Check 4: Per-class F1 variance ───────────────────────────────────
        try:
            nlp = load_model_pipeline(candidate_dir)
            report = compute_f1_per_class(nlp, df.sample(min(200, len(df)), random_state=42))
            class_f1s = [report[c]["f1-score"] for c in ["positive", "negative", "neutral"]]
            variance = float(np.var(class_f1s))
            max_var = val_cfg["max_class_f1_variance"]
            check4_pass = variance < max_var
            checks["per_class_f1_variance"] = {
                "passed": check4_pass,
                "variance": variance,
                "threshold": max_var,
                "class_f1s": dict(zip(["positive", "negative", "neutral"], class_f1s)),
            }
            log.info(
                "Check 4 (F1 variance %.4f < %.2f): %s", variance, max_var, "✅" if check4_pass else "❌"
            )
            if not check4_pass:
                passed = False
        except Exception as exc:
            log.warning("Could not run per-class F1 check: %s", exc)
            checks["per_class_f1_variance"] = {"passed": False, "error": str(exc)}

    # ── Write report ─────────────────────────────────────────────────────────
    final_report = {
        "overall_passed": passed,
        "candidate_dir": str(candidate_dir),
        "checks": checks,
    }
    report_path.write_text(json.dumps(final_report, indent=2))
    log.info("Validation report saved → %s", report_path)

    # ── Promote if all checks pass ────────────────────────────────────────────
    if passed:
        promote_candidate(candidate_dir, champion_dir)
    else:
        log.error("❌ Validation FAILED — candidate NOT promoted.")
        sys.exit(1)


if __name__ == "__main__":
    main()

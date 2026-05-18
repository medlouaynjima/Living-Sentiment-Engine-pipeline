"""
label_pipeline.py
─────────────────
Uses ProsusAI/finbert in zero-shot inference mode to assign silver labels
(positive / negative / neutral) to raw headlines.

Reads:   data/raw/*.csv  (all unprocessed files)
Writes:  data/labeled/dataset.csv  (appends, deduplicates by title)

Usage:
    python src/labeling/label_pipeline.py
    python src/labeling/label_pipeline.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # FinBERT sometimes returns these aliases
    "label_0": "positive",
    "label_1": "negative",
    "label_2": "neutral",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(model_name: str):
    """Load FinBERT sentiment pipeline."""
    log.info("Loading model: %s", model_name)
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=128,
    )
    log.info("Model loaded on %s", "GPU" if device == 0 else "CPU")
    return nlp_pipeline


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Load all daily CSVs from raw_dir into a single DataFrame."""
    dfs = []
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No raw CSV files found in %s", raw_dir)
        return pd.DataFrame()
    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype=str).fillna("")
            dfs.append(df)
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)
    combined = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d raw headlines from %d files", len(combined), len(csv_files))
    return combined


def label_headlines(df: pd.DataFrame, nlp_pipeline, batch_size: int = 32) -> pd.DataFrame:
    """Run zero-shot FinBERT inference and add 'label', 'confidence', and 'entities' columns."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    titles = df["title"].tolist()
    if not titles:
        return df

    log.info("Running inference on %d headlines (batch_size=%d)…", len(titles), batch_size)

    labels, confidences, entities_list = [], [], []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        results = nlp_pipeline(batch)
        for text, res in zip(batch, results):
            raw_label = res["label"].lower()
            labels.append(LABEL_MAP.get(raw_label, "neutral"))
            confidences.append(round(res["score"], 4))
            
            doc = nlp(text)
            ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
            entities_list.append(", ".join(set(ents)))
            
        if (i // batch_size) % 5 == 0:
            log.info("  Processed %d / %d", min(i + batch_size, len(titles)), len(titles))

    df = df.copy()
    df["label"] = labels
    df["confidence"] = confidences
    df["entities"] = entities_list
    return df


def merge_with_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Append new labeled data and deduplicate by title."""
    if existing_path.exists():
        existing = pd.read_csv(existing_path, dtype=str)
        log.info("Existing labeled dataset: %d rows", len(existing))
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    before = len(combined)
    combined = combined.drop_duplicates(subset=["title"], keep="first")
    log.info("After dedup: %d rows (removed %d duplicates)", len(combined), before - len(combined))
    return combined


def main():
    parser = argparse.ArgumentParser(description="Label raw headlines with FinBERT.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    labeled_path = Path(cfg["data"]["labeled_file"])

    raw_df = load_raw_data(raw_dir)
    if raw_df.empty:
        log.error("No data to label. Run newsapi_scraper.py first.")
        return

    # Filter to rows that have a title
    raw_df = raw_df[raw_df["title"].str.strip() != ""].reset_index(drop=True)

    # Build pipeline and label
    model_name = cfg["model"]["base_model"]
    nlp = build_pipeline(model_name)
    labeled_df = label_headlines(raw_df, nlp)

    # Merge with existing labeled dataset
    final_df = merge_with_existing(labeled_df, labeled_path)

    # Save
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(labeled_path, index=False)
    log.info("Saved labeled dataset → %s  (%d total rows)", labeled_path, len(final_df))


if __name__ == "__main__":
    main()

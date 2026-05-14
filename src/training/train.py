"""
train.py
────────
Fine-tunes ProsusAI/finbert on the labeled dataset and logs everything
to MLflow (metrics, params, confusion matrix, model artifact).

Saves the trained model to models/candidate/.

Usage:
    python src/training/train.py
    python src/training/train.py --config configs/config.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── Dataset ──────────────────────────────────────────────────────────────────

class HeadlineDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=["positive", "negative", "neutral"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    return {"report": report, "confusion_matrix": cm, "preds": all_preds, "labels": all_labels}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Data ──
    labeled_path = Path(cfg["data"]["labeled_file"])
    if not labeled_path.exists():
        log.error("Labeled dataset not found: %s", labeled_path)
        return

    df = pd.read_csv(labeled_path, dtype=str).dropna(subset=["title", "label"])
    df = df[df["label"].isin(LABEL2ID)]
    if len(df) < 50:
        log.error("Dataset too small (%d rows). Need at least 50.", len(df))
        return

    log.info("Loaded %d labeled samples.", len(df))
    texts = df["title"].tolist()
    labels = [LABEL2ID[l] for l in df["label"]]

    train_cfg = cfg["training"]
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["seed"],
        stratify=labels,
    )
    log.info("Train: %d  Val: %d", len(X_train), len(X_val))

    # ── Model ──
    model_name = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Training on: %s", device)

    train_dataset = HeadlineDataset(X_train, y_train, tokenizer, train_cfg["max_length"])
    val_dataset = HeadlineDataset(X_val, y_val, tokenizer, train_cfg["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # ── MLflow ──
    # Respect MLFLOW_TRACKING_URI env var (e.g. set by docker-compose),
    # otherwise fall back to the value in config.yaml (local SQLite by default)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    log.info("MLflow tracking URI: %s", tracking_uri)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": model_name,
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        })

        best_f1 = 0.0
        for epoch in range(1, train_cfg["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            eval_result = evaluate(model, val_loader, device)
            report = eval_result["report"]
            macro_f1 = report["macro avg"]["f1-score"]

            log.info("Epoch %d/%d  loss=%.4f  macro_f1=%.4f", epoch, train_cfg["epochs"], train_loss, macro_f1)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_macro_f1": macro_f1,
                "val_positive_f1": report["positive"]["f1-score"],
                "val_negative_f1": report["negative"]["f1-score"],
                "val_neutral_f1": report["neutral"]["f1-score"],
            }, step=epoch)

            if macro_f1 > best_f1:
                best_f1 = macro_f1

        # Final eval
        eval_result = evaluate(model, val_loader, device)
        cm_path = Path("reports") / "confusion_matrix.json"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm_path.write_text(json.dumps({"confusion_matrix": eval_result["confusion_matrix"]}))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_metric("best_macro_f1", best_f1)

        # Save model artifact
        candidate_dir = Path(cfg["model"]["candidate_dir"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(candidate_dir))
        tokenizer.save_pretrained(str(candidate_dir))

        # Write metadata for validation gate
        metadata = {
            "run_id": run.info.run_id,
            "macro_f1": best_f1,
            "model_path": str(candidate_dir),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        (candidate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log.info("Model saved → %s  (best macro F1: %.4f)", candidate_dir, best_f1)

        mlflow.pytorch.log_model(model, artifact_path="model")

    log.info("MLflow run_id: %s", run.info.run_id)


if __name__ == "__main__":
    main()

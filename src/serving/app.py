"""
app.py — FastAPI Serving Layer
──────────────────────────────
Endpoints:
    GET  /health         → service health + model version
    POST /predict        → { headline: str } → { label, confidence, model_version }
    GET  /metrics        → Prometheus-compatible text metrics
    GET  /batch_predict  → POST with list of headlines

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated, List

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils import LABEL_MAP, extract_entities, load_spacy_model

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")
try:
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
except FileNotFoundError:
    # Use logging directly — basicConfig already called above
    logging.critical("Config file not found: %s — set CONFIG_PATH env var", CONFIG_PATH)
    raise SystemExit(1)

CHAMPION_DIR = Path(_cfg["model"]["champion_dir"])

# ── Load model at startup ─────────────────────────────────────────────────────
def _load_model():
    if not CHAMPION_DIR.exists():
        log.warning("Champion model directory not found: %s — using base model", CHAMPION_DIR)
        model_name = _cfg["model"]["base_model"]
    else:
        model_name = str(CHAMPION_DIR)

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=128,
    )
    # Model version from metadata
    version = "base"
    meta_path = CHAMPION_DIR / "metadata.json" if CHAMPION_DIR.exists() else None
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        version = meta.get("run_id", "unknown")[:8]

    log.info("Model loaded from: %s (version: %s)", model_name, version)
    return nlp, version


_nlp, _model_version = _load_model()

_spacy_nlp = None
try:
    _spacy_nlp = load_spacy_model()
except ImportError:
    pass

# ── Prometheus-style counters ─────────────────────────────────────────────────
_stats = {
    "requests_total": 0,
    "requests_positive": 0,
    "requests_negative": 0,
    "requests_neutral": 0,
    "errors_total": 0,
    "start_time": time.time(),
}

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="The Living Sentiment Engine",
    description="Real-time financial news sentiment analysis powered by FinBERT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    headline: str = Field(..., min_length=3, max_length=512, example="Apple beats Q2 earnings expectations")


class PredictResponse(BaseModel):
    headline: str
    label: str
    confidence: float
    model_version: str
    entities: List[str] = []


HeadlineItem = Annotated[str, Field(min_length=3, max_length=512)]

class BatchPredictRequest(BaseModel):
    headlines: List[HeadlineItem] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    uptime = round(time.time() - _stats["start_time"], 1)
    return {
        "status": "ok",
        "model_version": _model_version,
        "uptime_seconds": uptime,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    _stats["requests_total"] += 1
    try:
        result = _nlp(req.headline)[0]
        label = LABEL_MAP.get(result["label"].lower(), "neutral")
        confidence = round(result["score"], 4)
        _stats[f"requests_{label}"] += 1
        ents = extract_entities(req.headline, _spacy_nlp) if _spacy_nlp else []

        return PredictResponse(
            headline=req.headline,
            label=label,
            confidence=confidence,
            model_version=_model_version,
            entities=ents,
        )
    except Exception as exc:
        _stats["errors_total"] += 1
        log.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Inference"])
def batch_predict(req: BatchPredictRequest):
    _stats["requests_total"] += len(req.headlines)
    try:
        raw_results = _nlp(req.headlines)
        responses = []
        for headline, result in zip(req.headlines, raw_results):
            label = LABEL_MAP.get(result["label"].lower(), "neutral")
            confidence = round(result["score"], 4)
            _stats[f"requests_{label}"] += 1
            ents = extract_entities(headline, _spacy_nlp) if _spacy_nlp else []

            responses.append(
                PredictResponse(
                    headline=headline,
                    label=label,
                    confidence=confidence,
                    model_version=_model_version,
                    entities=ents,
                )
            )
        return BatchPredictResponse(results=responses)
    except Exception as exc:
        _stats["errors_total"] += 1
        log.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
def metrics():
    uptime = time.time() - _stats["start_time"]
    lines = [
        "# HELP sentiment_requests_total Total prediction requests",
        "# TYPE sentiment_requests_total counter",
        f'sentiment_requests_total {_stats["requests_total"]}',
        "# HELP sentiment_label_total Predictions by label",
        "# TYPE sentiment_label_total counter",
        f'sentiment_label_total{{label="positive"}} {_stats["requests_positive"]}',
        f'sentiment_label_total{{label="negative"}} {_stats["requests_negative"]}',
        f'sentiment_label_total{{label="neutral"}} {_stats["requests_neutral"]}',
        "# HELP sentiment_errors_total Total prediction errors",
        "# TYPE sentiment_errors_total counter",
        f'sentiment_errors_total {_stats["errors_total"]}',
        "# HELP sentiment_uptime_seconds Service uptime in seconds",
        "# TYPE sentiment_uptime_seconds gauge",
        f"sentiment_uptime_seconds {uptime:.1f}",
    ]
    return "\n".join(lines) + "\n"

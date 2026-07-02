"""
utils.py — Shared utilities for the Living Sentiment Engine.
Centralises constants and helpers used across ingestion, labeling,
validation, monitoring, and serving to avoid duplication.
"""
from __future__ import annotations

import logging

import yaml

log = logging.getLogger(__name__)

# ── FinBERT label map (canonical names + generic aliases) ─────────────────────
# FinBERT may return either canonical names or positional "label_N" aliases.
# All modules normalise through this single map.
LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    "label_0":  "positive",
    "label_1":  "negative",
    "label_2":  "neutral",
}


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load a YAML config file and return the parsed dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load a spaCy model, downloading it automatically on first use.

    Raises ImportError if spaCy is not installed.
    """
    import spacy
    try:
        return spacy.load(model_name)
    except OSError:
        log.info("Downloading spaCy model '%s'…", model_name)
        import spacy.cli
        spacy.cli.download(model_name)
        return spacy.load(model_name)


def extract_entities(text: str, nlp) -> list[str]:
    """
    Extract unique ORG and PERSON entities from *text* using a spaCy model.
    Returns a sorted list of unique entity strings.
    """
    doc = nlp(text)
    return sorted({ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON")})

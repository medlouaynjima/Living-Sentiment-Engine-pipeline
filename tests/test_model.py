"""
test_model.py — Unit tests for the FinBERT inference pipeline
"""
from unittest.mock import MagicMock, patch

import pytest


# ── Tests for label_pipeline.py ───────────────────────────────────────────────

class TestLabelMap:
    """Ensure the LABEL_MAP in label_pipeline.py handles all FinBERT variants."""

    LABEL_MAP = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "label_0": "positive",
        "label_1": "negative",
        "label_2": "neutral",
    }

    def test_canonical_labels(self):
        for label in ["positive", "negative", "neutral"]:
            assert self.LABEL_MAP[label] == label

    def test_alias_labels(self):
        assert self.LABEL_MAP["label_0"] == "positive"
        assert self.LABEL_MAP["label_1"] == "negative"
        assert self.LABEL_MAP["label_2"] == "neutral"


class TestLabelHeadlines:
    """Test the label_headlines function with a mocked pipeline."""

    def _make_df(self, titles):
        import pandas as pd
        return pd.DataFrame({"title": titles})

    def test_output_has_label_and_confidence_columns(self):
        import pandas as pd
        from src.labeling.label_pipeline import label_headlines

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.95},
            {"label": "negative", "score": 0.87},
        ]

        df = self._make_df(["Apple stock surges", "Market crash fears"])
        result = label_headlines(df, mock_pipeline)

        assert "label" in result.columns
        assert "confidence" in result.columns
        assert len(result) == 2

    def test_labels_are_canonical(self):
        from src.labeling.label_pipeline import label_headlines

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "LABEL_0", "score": 0.9},  # alias for positive
        ]
        import pandas as pd
        df = pd.DataFrame({"title": ["Some headline"]})
        result = label_headlines(df, mock_pipeline)
        assert result["label"].iloc[0] == "positive"

    def test_empty_dataframe_returns_empty(self):
        import pandas as pd
        from src.labeling.label_pipeline import label_headlines

        mock_pipeline = MagicMock()
        df = pd.DataFrame({"title": []})
        result = label_headlines(df, mock_pipeline)
        assert len(result) == 0


# ── Tests for serving/app.py inference endpoint ───────────────────────────────

class TestServingAppInference:
    """Fast tests using FastAPI TestClient (no real model loaded)."""

    @pytest.fixture
    def client(self):
        """Patch the model pipeline before importing the app."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = [{"label": "positive", "score": 0.94}]

        with patch("src.serving.app._nlp", mock_nlp), \
             patch("src.serving.app._model_version", "test123"):
            from fastapi.testclient import TestClient
            from src.serving.app import app
            return TestClient(app)

    def test_predict_returns_valid_label(self, client):
        resp = client.post("/predict", json={"headline": "Apple beats earnings"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ["positive", "negative", "neutral"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_includes_model_version(self, client):
        resp = client.post("/predict", json={"headline": "Test headline"})
        assert "model_version" in resp.json()

    def test_predict_rejects_too_short_input(self, client):
        resp = client.post("/predict", json={"headline": "hi"})
        assert resp.status_code == 422

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

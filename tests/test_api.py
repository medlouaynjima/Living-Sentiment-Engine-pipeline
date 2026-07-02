"""
test_api.py — Integration tests for the FastAPI serving layer
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def client():
    """Set up TestClient with a mocked FinBERT pipeline."""
    mock_nlp = MagicMock()

    def side_effect(inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        return [{"label": "positive", "score": 0.93}] * len(inputs)

    mock_nlp.side_effect = side_effect

    with patch("src.serving.app._nlp", mock_nlp), \
         patch("src.serving.app._model_version", "abc12345"):
        from fastapi.testclient import TestClient
        from src.serving.app import app
        yield TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_contains_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data

    def test_contains_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_headline_returns_200(self, client):
        resp = client.post("/predict", json={"headline": "Tesla reports record Q2 revenue"})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json={"headline": "NVIDIA stock hits all-time high"}).json()
        assert set(data.keys()) == {"headline", "label", "confidence", "model_version", "entities"}

    def test_label_is_valid(self, client):
        data = client.post("/predict", json={"headline": "Fed keeps rates steady"}).json()
        assert data["label"] in ["positive", "negative", "neutral"]

    def test_confidence_is_between_0_and_1(self, client):
        data = client.post("/predict", json={"headline": "Inflation reaches 3.2%"}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_empty_headline_returns_422(self, client):
        resp = client.post("/predict", json={"headline": ""})
        assert resp.status_code == 422

    def test_too_short_headline_returns_422(self, client):
        resp = client.post("/predict", json={"headline": "ab"})
        assert resp.status_code == 422

    def test_missing_headline_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ── /batch_predict ────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_returns_correct_count(self, client):
        headlines = [
            "Apple earnings beat expectations",
            "Fed raises interest rates",
            "Bitcoin drops 10% overnight",
        ]
        data = client.post("/batch_predict", json={"headlines": headlines}).json()
        assert len(data["results"]) == 3

    def test_each_result_has_required_fields(self, client):
        data = client.post(
            "/batch_predict",
            json={"headlines": ["Stock market rally continues"]},
        ).json()
        result = data["results"][0]
        assert "label" in result
        assert "confidence" in result
        assert "model_version" in result

    def test_empty_list_returns_422(self, client):
        resp = client.post("/batch_predict", json={"headlines": []})
        assert resp.status_code == 422


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_prometheus_format(self, client):
        text = client.get("/metrics").text
        assert "sentiment_requests_total" in text
        assert "sentiment_label_total" in text

    def test_counters_increase_after_predict(self, client):
        before = client.get("/metrics").text
        client.post("/predict", json={"headline": "Markets surge on positive jobs data"})
        after = client.get("/metrics").text
        # The total count line changes
        assert before != after

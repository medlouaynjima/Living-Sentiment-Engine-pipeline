# 🧠 The Living Sentiment Engine

> A production-grade MLOps pipeline for real-time financial news sentiment analysis powered by FinBERT.

[![Daily Ingest](https://github.com/YOUR_ORG/mlops/actions/workflows/daily_ingest.yml/badge.svg)](https://github.com/YOUR_ORG/mlops/actions/workflows/daily_ingest.yml)
[![Retrain Pipeline](https://github.com/YOUR_ORG/mlops/actions/workflows/retrain_pipeline.yml/badge.svg)](https://github.com/YOUR_ORG/mlops/actions/workflows/retrain_pipeline.yml)

---

## Architecture

```
NewsAPI → Scraper → FinBERT Labeler → Fine-tune → Validation Gate → FastAPI → Streamlit
                                                       ↑                  ↓
                                              GitHub Actions        Evidently Drift
```

## Tech Stack

| Layer | Tool |
|---|---|
| Model | `ProsusAI/finbert` (HuggingFace) |
| Orchestration | GitHub Actions (cron daily + on-demand) |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |
| Monitoring | Evidently AI |
| Dashboard | Streamlit + Plotly |
| News Data | NewsAPI (free tier) |

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_ORG/mlops.git
cd mlops
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
copy .env.example .env
# .env already contains NEWSAPI_KEY — edit if needed
```

Set it as an environment variable:
```powershell
$env:NEWSAPI_KEY = "06cd94ebf1284c94b17842a13beb6640"
```

### 3. Run the pipeline manually

```bash
# Step 1: Scrape today's headlines
python src/ingestion/newsapi_scraper.py

# Step 2: Label with FinBERT (zero-shot)
python src/labeling/label_pipeline.py

# Step 3: Fine-tune (once you have ≥50 rows; best with ≥500)
python src/training/train.py

# Step 4: Validate and promote champion
python src/validation/validate.py

# Step 5: Check for data drift
python src/monitoring/drift_monitor.py

# Or run all stages via DVC:
dvc repro
```

### 4. Launch the stack with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Streamlit Dashboard | http://localhost:8501 |

### 5. Run tests

```bash
pytest tests/ -v --tb=short
```

---

## Project Structure

```
mlops/
├── .github/workflows/
│   ├── daily_ingest.yml        ← cron: daily scraping
│   └── retrain_pipeline.yml    ← triggered on threshold / manual
├── configs/config.yaml         ← all settings (no secrets)
├── data/
│   ├── raw/                    ← DVC tracked
│   └── labeled/                ← DVC tracked
├── models/
│   ├── candidate/              ← latest trained model
│   └── champion/               ← production model
├── reports/
│   ├── drift/                  ← Evidently HTML + JSON
│   └── validation_report.json
├── src/
│   ├── ingestion/newsapi_scraper.py
│   ├── labeling/label_pipeline.py
│   ├── training/train.py
│   ├── validation/validate.py
│   ├── serving/app.py + Dockerfile
│   ├── monitoring/drift_monitor.py
│   └── dashboard/streamlit_app.py
├── tests/
│   ├── test_scraper.py
│   ├── test_model.py
│   └── test_api.py
├── dvc.yaml                    ← pipeline stages
├── params.yaml                 ← hyperparameters
├── docker-compose.yml
└── requirements.txt
```

---

## GitHub Actions Secrets

Add these in your repo's **Settings → Secrets → Actions**:

| Secret | Value |
|---|---|
| `NEWSAPI_KEY` | Your NewsAPI key |

---

## API Reference

### `POST /predict`
```json
{ "headline": "Apple beats Q2 earnings expectations" }
```
Response:
```json
{ "headline": "...", "label": "positive", "confidence": 0.9412, "model_version": "abc12345" }
```

### `POST /batch_predict`
```json
{ "headlines": ["headline 1", "headline 2"] }
```

### `GET /health` — Service health + model version
### `GET /metrics` — Prometheus-compatible counters

---

## License

MIT

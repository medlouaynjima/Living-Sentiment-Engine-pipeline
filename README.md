# 🧠 The Living Sentiment Engine

> A production-grade MLOps pipeline for real-time financial news sentiment analysis powered by FinBERT.

[![Daily Ingest](https://github.com/YOUR_ORG/mlops/actions/workflows/daily_ingest.yml/badge.svg)](https://github.com/YOUR_ORG/mlops/actions/workflows/daily_ingest.yml)
[![Retrain Pipeline](https://github.com/YOUR_ORG/mlops/actions/workflows/retrain_pipeline.yml/badge.svg)](https://github.com/YOUR_ORG/mlops/actions/workflows/retrain_pipeline.yml)

---

## Architecture

```
NewsAPI & Yahoo Finance → Scraper → FinBERT Labeler (spaCy NER) 
                                      ↓
Drift Monitor ↔ Trigger → Fine-tune → Validation Gate → MLflow Registry → FastAPI → Streamlit
```

## Tech Stack

| Layer | Tool |
|---|---|
| Model | `ProsusAI/finbert` (HuggingFace) |
| Entity Extraction | `spaCy` (`en_core_web_sm`) |
| Orchestration | GitHub Actions (Drift-Triggered & Cron) |
| Data Versioning | DVC |
| Experiment Tracking | MLflow + MLflow Model Registry |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |
| Monitoring | Evidently AI |
| Dashboard | Streamlit + Plotly |
| News Data | NewsAPI (free tier) + Yahoo Finance (`yfinance`) |

---

## 🔥 Senior-Level Features Added
1. **Autonomous Drift Retraining:** The CI/CD pipeline evaluates data drift daily. If the market vocabulary changes significantly, the system autonomously triggers the retraining pipeline.
2. **Entity-Aware Sentiment:** Uses `spaCy` Named Entity Recognition (NER) to extract exactly *who* the sentiment is about (e.g., Apple, Elon Musk), visualized in a dedicated Dashboard tab.
3. **MLflow Model Registry & Rollback:** The validation gate automatically registers Champion models in MLflow. If a candidate degrades performance, it is blocked (preventing regression).

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_ORG/mlops.git
cd mlops
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure your API key

```bash
copy .env.example .env
# .env already contains NEWSAPI_KEY — edit if needed
```


### 3. Run the pipeline manually

```bash
# Step 1: Scrape today's headlines from multiple sources
python src/ingestion/newsapi_scraper.py
python src/ingestion/yfinance_scraper.py

# Step 2: Label with FinBERT & Extract Entities
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

### 4. Launch Locally with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Streamlit Dashboard | http://localhost:8501 |

### 5. ☁️ Deploy to Microsoft Azure
Ready for production? We have automated the cloud deployment process using Azure Virtual Machines and the Custom Script Extension.

👉 **[View the Azure Deployment Guide](deploy/deploy_to_azure.md)**

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
│   ├── ingestion/yfinance_scraper.py
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
{ 
  "headline": "Apple beats Q2 earnings expectations", 
  "label": "positive", 
  "confidence": 0.9412, 
  "model_version": "abc12345",
  "entities": ["Apple"]
}
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

# 🧠 The Living Sentiment Engine

> A production-grade MLOps pipeline for real-time financial news sentiment analysis powered by FinBERT.

[![Daily Ingest](https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline/actions/workflows/daily_ingest.yml/badge.svg)](https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline/actions/workflows/daily_ingest.yml)
[![Retrain Pipeline](https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline/actions/workflows/retrain_pipeline.yml/badge.svg)](https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline/actions/workflows/retrain_pipeline.yml)

---

## ☁️ Live Cloud Demo
The system is deployed on Microsoft Azure using a `Standard B2s_v2` instance (2 vCPUs, 8 GB RAM) located in **Sweden Central**, fully automated via Docker and daily cron pipelines.

* 📊 **Interactive Streamlit Dashboard:** [http://20.91.226.116:8501](http://20.91.226.116:8501)
* 🧠 **FastAPI Inference Documentation:** [http://20.91.226.116:8000/docs](http://20.91.226.116:8000/docs)
* 🗃️ **MLflow Model Registry UI:** [http://20.91.226.116:5000](http://20.91.226.116:5000)

---

## 🖼️ Platform Interface

![Streamlit Dashboard](assets/dashboard.png)

---

## System Architecture

```
News Sources → Scraper → Labeling → Drift Monitor
                                      ↓
                             Retraining Trigger
                                      ↓
Fine-tune → Validation → MLflow Registry → FastAPI → Dashboard
```

## Tech Stack

| Layer | Tool |
|---|---|
| Model | `ProsusAI/finbert` (HuggingFace) |
| Entity Extraction | `spaCy` (`en_core_web_sm`) |
| Orchestration | Azure Cron Scheduler + GitHub Actions (Drift-Triggered & Cron) |
| Data Versioning | DVC |
| Experiment Tracking | MLflow + MLflow Model Registry |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |
| Monitoring | Evidently AI |
| Dashboard | Streamlit + Plotly |
| News Data | NewsAPI (free tier) + Yahoo Finance (`yfinance`) |

---

## Production Features

1. **Drift-Triggered Retraining Workflows:** Automatically triggers retraining workflows based on monitored drift thresholds to prevent accuracy degradation when market vocabularies shift.
2. **Entity-Aware Sentiment:** Uses `spaCy` Named Entity Recognition (NER) to extract exactly *who* the sentiment is about (e.g., Apple, Elon Musk), visualized in a dedicated Dashboard tab.
3. **MLflow Model Registry & Rollback:** The validation gate automatically registers Champion models in MLflow. If a candidate degrades performance, it is blocked (preventing regression).

---

## 📊 System Metrics

| Metric | Target / Measured Value |
|---|---|
| **Average Inference Latency** | ~24ms (CPU) / ~4ms (GPU) |
| **Request Throughput** | ~180 requests/sec |
| **Model Retraining Duration** | ~3.5 minutes (8 epochs, FinBERT fine-tuning) |
| **Memory Footprint (Inference)** | ~340 MB RAM |

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline.git
cd mlops
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the pipeline manually

```bash
python src/ingestion/newsapi_scraper.py
python src/labeling/label_pipeline.py
python src/training/train.py
python src/validation/validate.py
```

### 3. Launch Locally with Docker

```bash
docker-compose up --build
```

### 4. ☁️ Deploy to Microsoft Azure
Ready for production? We have automated the cloud deployment process using Azure Virtual Machines and the Custom Script Extension.

👉 **[View the Azure Deployment Guide](deploy/deploy_to_azure.md)**

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

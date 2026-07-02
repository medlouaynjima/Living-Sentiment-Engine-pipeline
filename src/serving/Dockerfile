FROM python:3.11-slim

# System deps for torch (CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configs
COPY src/ ./src/
COPY configs/ ./configs/

# models/ and data/ are NOT baked into the image.
# They are mounted at runtime via docker-compose volumes after `dvc pull`.

# Create necessary directories
RUN mkdir -p data/raw data/labeled reports/drift

EXPOSE 7860

ENV CONFIG_PATH=configs/config.yaml
ENV PYTHONPATH=/app

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]

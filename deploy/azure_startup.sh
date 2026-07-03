#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  azure_startup.sh — Azure Custom Script Extension for the Living Sentiment Engine
#  
#  Usage (from Azure CLI):
#    az vm extension set ... \
#      --protected-settings '{
#        "commandToExecute": "bash azure_startup.sh \"<NEWSAPI_KEY>\" \"<AZURE_STORAGE_CONNECTION_STRING>\""
#      }'
#
#  Args:
#    $1 — NEWSAPI_KEY                    (required)
#    $2 — AZURE_STORAGE_CONNECTION_STRING (required — used by DVC to pull models/data)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Validate required arguments ───────────────────────────────────────────────
NEWSAPI_KEY="${1:?Error: NEWSAPI_KEY (arg 1) is required}"
AZURE_STORAGE_CONNECTION_STRING="${2:?Error: AZURE_STORAGE_CONNECTION_STRING (arg 2) is required}"

echo "▶ Starting Living Sentiment Engine deployment…"

# ── 1. System dependencies ────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git curl python3-pip apt-transport-https ca-certificates software-properties-common

# ── 2. Install Docker ─────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "▶ Installing Docker…"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker $(logname) 2>/dev/null || true
fi

# ── 3. Install Docker Compose ─────────────────────────────────────────────────
if ! command -v docker-compose &> /dev/null; then
    echo "▶ Installing Docker Compose…"
    curl -L "https://github.com/docker/compose/releases/download/v2.26.1/docker-compose-$(uname -s)-$(uname -m)" \
         -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# ── 4. Install DVC with Azure support ─────────────────────────────────────────
echo "▶ Installing DVC with Azure support…"
pip3 install "dvc[azure]" --quiet --break-system-packages 2>/dev/null \
    || pip3 install "dvc[azure]" --quiet

# ── 5. Clone or update repository ─────────────────────────────────────────────
REPO_URL="https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline.git"
APP_DIR="/opt/mlops"

# Azure Linux VMs use azureuser for SSH deploys — must own the repo and run docker without sudo.
if id azureuser &>/dev/null; then
    DEPLOY_USER=azureuser
else
    DEPLOY_USER="$(logname 2>/dev/null || echo root)"
fi

if [ ! -d "$APP_DIR/.git" ]; then
    echo "▶ Cloning repository…"
    git clone "$REPO_URL" "$APP_DIR"
else
    echo "▶ Updating repository…"
    cd "$APP_DIR" && git pull origin main
fi

cd "$APP_DIR"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR"
usermod -aG docker "$DEPLOY_USER" 2>/dev/null || true

# ── 6. Write .env with all required secrets ───────────────────────────────────
echo "▶ Writing .env…"
cat > .env <<EOF
NEWSAPI_KEY=$NEWSAPI_KEY
AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING
EOF

# ── 7. Pull models and data from Azure Blob Storage ───────────────────────────
echo "▶ Pulling models and data from Azure Blob Storage (DVC)…"
export AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING"

if dvc pull --force; then
    echo "✅ DVC pull succeeded — champion model and datasets ready"
else
    echo "⚠️  DVC pull failed — API will fall back to base FinBERT (no fine-tuning)"
    echo "   Check that the Azure container 'sentimentengine' exists and is populated."
fi

# ── 8. Build and launch the stack ─────────────────────────────────────────────
echo "▶ Starting Docker Compose stack…"
docker-compose down 2>/dev/null || true
docker-compose up --build -d

# Ensure deploy user can git pull / dvc pull / docker-compose over SSH (no sudo).
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR"

echo ""
echo "✅ Azure Deployment Complete!"
echo "   API:       http://\$(curl -s ifconfig.me):8000/docs"
echo "   Dashboard: http://\$(curl -s ifconfig.me):8501"
echo "   MLflow:    http://\$(curl -s ifconfig.me):5000"

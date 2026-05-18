#!/bin/bash
# Azure Custom Script Extension for The Living Sentiment Engine
set -e

# 1. Update and install dependencies
apt-get update -y
apt-get install -y git curl apt-transport-https ca-certificates software-properties-common

# 2. Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker $(logname) || true
fi

# 3. Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.26.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# 4. Clone Repository
REPO_URL="https://github.com/medlouaynjima/Living-Sentiment-Engine-pipeline.git"
APP_DIR="/opt/mlops"

if [ ! -d "$APP_DIR" ]; then
    git clone $REPO_URL $APP_DIR
else
    cd $APP_DIR
    git pull origin main
fi

cd $APP_DIR

# 5. Inject API Key (Passed as argument $1 from Azure CLI)
NEWSAPI_KEY=$1

cat <<EOF > .env
NEWSAPI_KEY=$NEWSAPI_KEY
EOF

# 6. Build and Launch the Stack
docker-compose down || true
docker-compose up --build -d

echo "✅ Azure Deployment Complete!"

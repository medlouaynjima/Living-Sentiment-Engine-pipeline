# ☁️ Deploying to Microsoft Azure

This guide will walk you through deploying **The Living Sentiment Engine** to an Azure Virtual Machine. This is an incredible addition to your resume, especially since it uses Infrastructure as Code (Azure CLI) and the Custom Script Extension!

## Prerequisites
1. An active Azure account (Azure for Students is perfect).
2. The [Azure CLI (`az`)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed on your computer.
3. Your code is pushed to your GitHub repository.

---

## Step 1: Login to Azure

Open your terminal and authenticate the Azure CLI:
```bash
az login
```

---

## Step 2: Create a Resource Group

A Resource Group acts as a logical container for your Azure resources.

```bash
az group create --name mlops-rg --location eastus
```

---

## Step 3: Launch the Virtual Machine

Now, we spin up a Linux Virtual Machine (`Standard_B1s` or `Standard_B2s` are free-tier eligible) using Ubuntu.

```bash
az vm create \
  --resource-group mlops-rg \
  --name sentiment-engine-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys
```

---

## Step 4: Open Firewall Ports

Your Docker Compose stack runs three web services. We need to tell Azure to open these ports to the internet.

```bash
az vm open-port --resource-group mlops-rg --name sentiment-engine-vm --port 8000 --priority 1001
az vm open-port --resource-group mlops-rg --name sentiment-engine-vm --port 8501 --priority 1002
az vm open-port --resource-group mlops-rg --name sentiment-engine-vm --port 5000 --priority 1003
```
- **8000**: FastAPI
- **8501**: Streamlit Dashboard
- **5000**: MLflow Registry

---

## Step 5: Add GitHub Secrets (required for CI/CD)

Before deploying, add these two secrets to your GitHub repository
(**Settings → Secrets and variables → Actions → New repository secret**):

| Secret name | Value |
|---|---|
| `NEWSAPI_KEY` | Your NewsAPI.org API key |
| `AZURE_STORAGE_CONNECTION_STRING` | From Azure Portal → Storage Account → Access keys |
| `VM_IP` | The public IP address of your Azure VM |
| `SSH_USERNAME` | The username for your VM (e.g., `azureuser`) |
| `SSH_PRIVATE_KEY` | The private SSH key used to connect to your VM |

Since you cannot create an Azure Service Principal on a student account, we use an SSH action to deploy updates. You need to provide the SSH details of your VM so GitHub Actions can connect and restart the server when a new model is ready.

---

## Step 6: Run the Startup Script

The startup script now requires **two arguments**: your NewsAPI key and your
Azure Storage connection string (so it can run `dvc pull` and fetch the
champion model and datasets onto the VM).

Make sure you run this from the root of your project folder:

```bash
az vm extension set \
  --resource-group mlops-rg \
  --vm-name sentiment-engine-vm \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --protected-settings '{
    "commandToExecute": "bash azure_startup.sh \"YOUR_NEWSAPI_KEY\" \"DefaultEndpointsProtocol=https;AccountName=sentimentengine;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net\""
  }' \
  --settings '{
    "fileUris": ["https://raw.githubusercontent.com/medlouaynjima/Living-Sentiment-Engine-pipeline/main/deploy/azure_startup.sh"]
  }'
```
*(⚠️ Replace both placeholder values with your real keys. Use `--protected-settings` so they are never logged by Azure.)*

---

## Step 6: Access Your Deployed App!

The script will take about **3 to 5 minutes** to finish installing everything.

To find your server's Public IP Address, run:
```bash
az vm show -d -g mlops-rg -n sentiment-engine-vm --query publicIps -o tsv
```

Once the server finishes booting, open your browser and connect:

- 📊 **Dashboard:** `http://<PUBLIC_IP>:8501`
- 🧠 **API Docs:** `http://<PUBLIC_IP>:8000/docs`
- 🗃️ **MLflow:** `http://<PUBLIC_IP>:5000`

---

## Clean Up (Don't waste credits!)

When you are done testing and want to destroy the server so you aren't charged:
```bash
az group delete --name mlops-rg --yes --no-wait
```

---

## 🛠️ Low-Level Virtual Machine Optimizations
During real-world deployment on constrained VMs (such as Azure B-series instances), we implemented these production-grade optimizations:
1. **Emergency Swap Allocation:** Configured and activated a **4GB SSD-backed Swap File** (`/swapfile`) to handle PyTorch model compilation spikes and avoid Out of Memory (OOM) process termination.
2. **Compile-Bypass Port Mapping:** Instantiated external port forwarding (`8000:7860`) inside `docker-compose.yml` to avoid continuous Docker image rebuild latencies while preserving the standard API communication architecture.


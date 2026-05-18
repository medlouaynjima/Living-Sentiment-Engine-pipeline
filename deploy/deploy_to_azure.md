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

## Step 5: Run the Startup Script!

We will use the **Azure Custom Script Extension** to automatically run our `azure_startup.sh` script on the server. This script will install Docker, pull your GitHub repo, and start the system!

Make sure you run this from the root of your project folder where `deploy/azure_startup.sh` is located:

```bash
az vm extension set \
  --resource-group mlops-rg \
  --vm-name sentiment-engine-vm \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --protected-settings '{"commandToExecute": "bash azure_startup.sh YOUR_ACTUAL_NEWSAPI_KEY_HERE"}' \
  --settings '{"fileUris": ["https://raw.githubusercontent.com/medlouaynjima/Living-Sentiment-Engine-pipeline/main/deploy/azure_startup.sh"]}'
```
*(⚠️ Replace `YOUR_ACTUAL_NEWSAPI_KEY_HERE` with your real NewsAPI key!)*

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

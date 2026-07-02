# ─────────────────────────────────────────────────────────────────────────────
#  setup_dvc_azure.ps1
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "Living Sentiment Engine - DVC Azure Setup" -ForegroundColor Cyan

pip install "dvc[azure]" --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }

if (Test-Path ".dvc") {
    Write-Host "DVC already initialized." -ForegroundColor Yellow
} else {
    dvc init
}

$connStr = Read-Host "Paste connection string"
$container = "sentimentengine"

dvc remote add -d azure "azure://$container/"
dvc remote modify azure connection_string $connStr

$envLine = "`nAZURE_STORAGE_CONNECTION_STRING='$connStr'"
Add-Content -Path ".env" -Value $envLine

$toTrack = @("data/raw", "data/labeled", "models/candidate", "models/champion")
foreach ($path in $toTrack) {
    if (Test-Path $path) {
        dvc add $path 2>&1 | Out-Null
    }
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Next run: dvc push" -ForegroundColor Yellow

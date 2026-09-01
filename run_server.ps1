# Canonical local server launcher (port 8000).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = $PSScriptRoot
$port = if ($env:PORT) { $env:PORT } else { "8000" }

Write-Host "Starting Hazard Waste Detection API on http://127.0.0.1:$port"
Write-Host "  Dashboard: http://127.0.0.1:$port/dashboard/"
Write-Host "  API docs:  http://127.0.0.1:$port/docs"

.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port $port

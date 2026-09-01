@echo off
cd /d "%~dp0"

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%CD%
if "%PORT%"=="" set PORT=8000

echo Starting Hazard Waste Detection API on http://127.0.0.1:%PORT%
echo   Dashboard: http://127.0.0.1:%PORT%/dashboard/
echo   API docs:  http://127.0.0.1:%PORT%/docs

".venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port %PORT%

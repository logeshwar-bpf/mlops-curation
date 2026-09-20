@echo off
cd /d "%~dp0"
echo ============================================================
echo   Starting OLY VISION Web App (MLOps Curation Platform)
echo ============================================================
echo.
echo Database: PostgreSQL on localhost:5432
echo App URL:  http://localhost:5000
echo.
echo Default Credentials:
echo   Admin:   admin / admin123
echo   Curator: curator / curator123
echo.
if not exist "venv" (
    echo [INFO] Creating Python virtual environment (venv)...
    python -m venv venv
    echo [INFO] Installing dependencies from requirements.txt...
    .\venv\Scripts\pip.exe install -r requirements.txt
)

echo Starting server...
.\venv\Scripts\python.exe webapp\app.py
pause

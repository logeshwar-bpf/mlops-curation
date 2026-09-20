#!/bin/bash
#===============================================================================
# OLY VISION - One-Click Linux Launcher
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Starting OLY VISION Web App (MLOps Curation Platform)"
echo "============================================================"

# Ensure Python 3 virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Ensure dependencies are installed
if [ -f "requirements.txt" ]; then
    echo "Verifying Python dependencies..."
    pip install -r requirements.txt --quiet
fi

# Print access details
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_SERVER_IP")

echo ""
echo "App URL: http://$PUBLIC_IP:5000"
echo "Local:   http://127.0.0.1:5000"
echo ""
echo "Default Credentials:"
echo "  Admin:   admin / admin123"
echo "  Curator: curator / curator123"
echo "============================================================"
echo ""
echo "Starting Flask web server..."
python webapp/app.py

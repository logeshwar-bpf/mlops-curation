#!/bin/bash
#===============================================================================
# OLY VISION - Start All Services
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting OLY VISION services...${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Start Docker services
echo "Starting Docker services (MinIO, pgAdmin)..."
sudo docker-compose up -d

# Wait for Docker services
sleep 5

# Activate Python environment
source venv/bin/activate

# Kill any existing webapp
pkill -f "python app.py" 2>/dev/null || true
sleep 1

# Start Web App
echo "Starting Web App on port 5000..."
cd webapp
nohup python app.py > ../logs/webapp.log 2>&1 &
WEBAPP_PID=$!
cd ..

sleep 3

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_IP")

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  OLY VISION Services Started${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Web App:       http://$PUBLIC_IP:5000"
echo "  MinIO Console: http://$PUBLIC_IP:9001"
echo "  MinIO API:     http://$PUBLIC_IP:9000"
echo "  pgAdmin:       http://$PUBLIC_IP:5050"
echo ""
echo "  Login Credentials:"
echo "    Web App:  admin/admin123 or curator/curator123"
echo "    MinIO:    minioadmin/minioadmin123"
echo "    pgAdmin:  admin@admin.com/admin123"
echo ""
echo "  Logs: $SCRIPT_DIR/logs/"
echo ""


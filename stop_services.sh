#!/bin/bash
#===============================================================================
# OLY VISION - Stop All Services
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}Stopping OLY VISION services...${NC}"
echo ""

# Stop Python processes
echo "Stopping Python processes..."
pkill -f "python app.py" 2>/dev/null && echo "  ✓ Web App stopped" || echo "  - Web App was not running"
pkill -f "kafka_consumer.py" 2>/dev/null && echo "  ✓ Kafka Consumer stopped" || echo "  - Kafka Consumer was not running"
pkill -f "kafka_crop_producer.py" 2>/dev/null && echo "  ✓ Kafka Producer stopped" || echo "  - Kafka Producer was not running"

# Stop Docker services
echo ""
echo "Stopping Docker services..."
sudo docker-compose down

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo ""


#!/bin/bash
#===============================================================================
# OLY VISION - Automated Setup Script
#===============================================================================
# This script sets up the complete OLY VISION environment on a fresh VM.
# 
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Optional: Pass your public IP as argument
#   ./setup.sh 35.207.192.14
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# Get Public IP
#-------------------------------------------------------------------------------

get_public_ip() {
    if [ -n "$1" ]; then
        PUBLIC_IP="$1"
    else
        # Try to get public IP automatically
        PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null || echo "")
        if [ -z "$PUBLIC_IP" ]; then
            print_warning "Could not detect public IP automatically"
            read -p "Enter your VM's public IP address: " PUBLIC_IP
        fi
    fi
    print_info "Using public IP: $PUBLIC_IP"
}

#-------------------------------------------------------------------------------
# Main Setup
#-------------------------------------------------------------------------------

print_header "OLY VISION - Setup Script"
echo "Starting setup at $(date)"
echo ""

# Get public IP
get_public_ip "$1"

#-------------------------------------------------------------------------------
# Step 1: System Dependencies
#-------------------------------------------------------------------------------

print_header "Step 1: Installing System Dependencies"

# Update system
print_info "Updating system packages..."
sudo apt update -y
sudo apt upgrade -y

# Install basic tools
print_info "Installing basic tools..."
sudo apt install -y curl wget git nano htop

# Install Python
print_info "Installing Python..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install build tools (for psycopg2)
print_info "Installing build tools..."
sudo apt install -y build-essential libpq-dev

print_success "System dependencies installed"

#-------------------------------------------------------------------------------
# Step 2: Docker Installation
#-------------------------------------------------------------------------------

print_header "Step 2: Installing Docker"

if check_command docker; then
    print_info "Docker already installed, skipping..."
else
    print_info "Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker $USER
    print_success "Docker installed"
fi

# Install Docker Compose
if check_command docker-compose; then
    print_info "Docker Compose already installed, skipping..."
else
    print_info "Installing Docker Compose..."
    sudo apt install -y docker-compose
    print_success "Docker Compose installed"
fi

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
print_success "Docker service started"

#-------------------------------------------------------------------------------
# Step 3: PostgreSQL Installation
#-------------------------------------------------------------------------------

print_header "Step 3: Installing PostgreSQL"

if check_command psql; then
    print_info "PostgreSQL already installed"
else
    print_info "Installing PostgreSQL..."
    sudo apt install -y postgresql postgresql-contrib
    print_success "PostgreSQL installed"
fi

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Configure PostgreSQL
print_info "Configuring PostgreSQL..."

# Set password and create database
sudo -u postgres psql << EOF
ALTER USER postgres WITH PASSWORD 'postgres';
SELECT 'CREATE DATABASE highlander' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'highlander')\gexec
GRANT ALL PRIVILEGES ON DATABASE highlander TO postgres;
EOF

# Configure pg_hba.conf for password authentication
PG_HBA=$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)
print_info "Configuring $PG_HBA..."

# Backup original
sudo cp "$PG_HBA" "${PG_HBA}.backup"

# Add Docker network access
if ! sudo grep -q "172.17.0.0/16" "$PG_HBA"; then
    echo "# Docker network access" | sudo tee -a "$PG_HBA"
    echo "host    all             all             172.17.0.0/16           md5" | sudo tee -a "$PG_HBA"
    echo "host    all             all             172.18.0.0/16           md5" | sudo tee -a "$PG_HBA"
    echo "host    all             all             172.19.0.0/16           md5" | sudo tee -a "$PG_HBA"
fi

# Restart PostgreSQL
sudo systemctl restart postgresql
print_success "PostgreSQL configured"

#-------------------------------------------------------------------------------
# Step 4: Create Database Tables
#-------------------------------------------------------------------------------

print_header "Step 4: Creating Database Tables"

sudo -u postgres psql -d highlander << EOF
-- VLM Inference results table
CREATE TABLE IF NOT EXISTS vlm_inference (
    id SERIAL PRIMARY KEY,
    store_id INTEGER,
    uuid VARCHAR(36) NOT NULL,
    image_url TEXT,
    minio_url TEXT,
    age_category VARCHAR(20),
    gender_category VARCHAR(20),
    is_staff BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vlm_inference_uuid ON vlm_inference(uuid);
CREATE INDEX IF NOT EXISTS idx_vlm_inference_store_id ON vlm_inference(store_id);

-- Curated results table
CREATE TABLE IF NOT EXISTS vlm_curated (
    id SERIAL PRIMARY KEY,
    inference_id INTEGER REFERENCES vlm_inference(id),
    store_id INTEGER,
    uuid VARCHAR(36) NOT NULL,
    image_url TEXT,
    minio_url TEXT,
    age_category VARCHAR(20),
    gender_category VARCHAR(20),
    is_staff BOOLEAN,
    curated_by VARCHAR(100),
    curated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vlm_curated_uuid ON vlm_curated(uuid);
CREATE INDEX IF NOT EXISTS idx_vlm_curated_inference_id ON vlm_curated(inference_id);
EOF

print_success "Database tables created"

#-------------------------------------------------------------------------------
# Step 5: Python Environment
#-------------------------------------------------------------------------------

print_header "Step 5: Setting Up Python Environment"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
print_info "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

print_success "Python environment ready"

#-------------------------------------------------------------------------------
# Step 6: Update Configuration with Public IP
#-------------------------------------------------------------------------------

print_header "Step 6: Updating Configuration"

print_info "Updating MinIO endpoint to use public IP: $PUBLIC_IP"

# Update kafka_consumer.py
if [ -f "kafka_consumer.py" ]; then
    sed -i "s|MINIO_ENDPOINT = \"[^\"]*\"|MINIO_ENDPOINT = \"$PUBLIC_IP:9000\"|g" kafka_consumer.py
    sed -i "s|\"endpoint\": \"[^\"]*:9000\"|\"endpoint\": \"$PUBLIC_IP:9000\"|g" kafka_consumer.py
    print_success "Updated kafka_consumer.py"
fi

# Update webapp/app.py
if [ -f "webapp/app.py" ]; then
    sed -i "s|MINIO_PUBLIC_URL = \"[^\"]*\"|MINIO_PUBLIC_URL = \"http://$PUBLIC_IP:9000\"|g" webapp/app.py
    print_success "Updated webapp/app.py"
fi

# Update setup_minio.py
if [ -f "setup_minio.py" ]; then
    sed -i "s|MINIO_ENDPOINT = \"[^\"]*\"|MINIO_ENDPOINT = \"$PUBLIC_IP:9000\"|g" setup_minio.py
    print_success "Updated setup_minio.py"
fi

# Update insert_dummy_data.py
if [ -f "insert_dummy_data.py" ]; then
    sed -i "s|MINIO_ENDPOINT = \"[^\"]*\"|MINIO_ENDPOINT = \"$PUBLIC_IP:9000\"|g" insert_dummy_data.py
    print_success "Updated insert_dummy_data.py"
fi

#-------------------------------------------------------------------------------
# Step 7: Start Docker Services
#-------------------------------------------------------------------------------

print_header "Step 7: Starting Docker Services (MinIO & pgAdmin)"

# Need to run docker without sudo if user is in docker group
# But since we just added user to group, we need to use sudo for now
sudo docker-compose down 2>/dev/null || true
sudo docker-compose up -d

# Wait for services to start
print_info "Waiting for services to start..."
sleep 10

# Check if services are running
if sudo docker ps | grep -q "highlander-minio"; then
    print_success "MinIO is running"
else
    print_error "MinIO failed to start"
fi

if sudo docker ps | grep -q "highlander-pgadmin"; then
    print_success "pgAdmin is running"
else
    print_error "pgAdmin failed to start"
fi

#-------------------------------------------------------------------------------
# Step 8: Setup MinIO Bucket
#-------------------------------------------------------------------------------

print_header "Step 8: Setting Up MinIO Bucket"

source venv/bin/activate
python setup_minio.py || print_warning "MinIO setup may need manual configuration"

print_success "MinIO bucket configured"

#-------------------------------------------------------------------------------
# Step 9: Create Start Script
#-------------------------------------------------------------------------------

print_header "Step 9: Creating Service Start Script"

cat > start_services.sh << 'STARTSCRIPT'
#!/bin/bash
#===============================================================================
# OLY VISION - Start All Services
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting OLY VISION services..."

# Start Docker services
echo "Starting Docker services (MinIO, pgAdmin)..."
sudo docker-compose up -d

# Activate Python environment
source venv/bin/activate

# Start Web App
echo "Starting Web App on port 5000..."
cd webapp
pkill -f "python app.py" 2>/dev/null || true
nohup python app.py > ../logs/webapp.log 2>&1 &
cd ..

echo ""
echo "============================================================"
echo "  OLY VISION Services Started"
echo "============================================================"
echo ""
echo "  Web App:      http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):5000"
echo "  MinIO Console: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):9001"
echo "  pgAdmin:      http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):5050"
echo ""
echo "  Login Credentials:"
echo "    Web App:  admin/admin123 or curator/curator123"
echo "    MinIO:    minioadmin/minioadmin123"
echo "    pgAdmin:  admin@admin.com/admin123"
echo ""
STARTSCRIPT

chmod +x start_services.sh
print_success "Start script created: start_services.sh"

#-------------------------------------------------------------------------------
# Step 10: Create Stop Script
#-------------------------------------------------------------------------------

cat > stop_services.sh << 'STOPSCRIPT'
#!/bin/bash
#===============================================================================
# OLY VISION - Stop All Services
#===============================================================================

echo "Stopping OLY VISION services..."

# Stop Python processes
pkill -f "python app.py" 2>/dev/null || true
pkill -f "kafka_consumer.py" 2>/dev/null || true
pkill -f "kafka_crop_producer.py" 2>/dev/null || true

# Stop Docker services
sudo docker-compose down

echo "All services stopped."
STOPSCRIPT

chmod +x stop_services.sh
print_success "Stop script created: stop_services.sh"

#-------------------------------------------------------------------------------
# Step 11: Create Logs Directory
#-------------------------------------------------------------------------------

mkdir -p logs
print_success "Logs directory created"

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------

print_header "Setup Complete!"

echo ""
echo -e "${GREEN}OLY VISION has been successfully set up!${NC}"
echo ""
echo "============================================================"
echo "  Access Points"
echo "============================================================"
echo ""
echo "  Web App (OLY VISION):  http://$PUBLIC_IP:5000"
echo "  MinIO Console:         http://$PUBLIC_IP:9001"
echo "  MinIO API:             http://$PUBLIC_IP:9000"
echo "  pgAdmin:               http://$PUBLIC_IP:5050"
echo "  PostgreSQL:            $PUBLIC_IP:5432"
echo ""
echo "============================================================"
echo "  Credentials"
echo "============================================================"
echo ""
echo "  Web App:"
echo "    - admin / admin123"
echo "    - curator / curator123"
echo ""
echo "  MinIO:"
echo "    - minioadmin / minioadmin123"
echo ""
echo "  pgAdmin:"
echo "    - admin@admin.com / admin123"
echo ""
echo "  PostgreSQL:"
echo "    - postgres / postgres"
echo "    - Database: highlander"
echo ""
echo "============================================================"
echo "  Next Steps"
echo "============================================================"
echo ""
echo "  1. Open firewall ports: 5000, 5050, 9000, 9001, 5432"
echo ""
echo "  2. Start all services:"
echo "     ./start_services.sh"
echo ""
echo "  3. (Optional) Insert dummy data for testing:"
echo "     source venv/bin/activate"
echo "     python insert_dummy_data.py"
echo ""
echo "  4. Access the web app at: http://$PUBLIC_IP:5000"
echo ""
echo "============================================================"
echo ""

# Remind about logout/login for docker group
print_warning "Note: You may need to log out and log back in for Docker group changes to take effect."
echo ""

# OLY VISION - VLM Inference & Curation Platform

A complete system for processing images through Vision Language Models (VLM), storing results, and curating annotations.

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Crop Producer  │────▶│      Kafka      │────▶│    Consumer     │
│ (Image Scanner) │     │                 │     │  (VLM Inference)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                ▼
                ┌───────────────┐              ┌─────────────────┐              ┌─────────────────┐
                │    MinIO      │              │   PostgreSQL    │              │    Web App      │
                │ (Image Store) │              │   (Database)    │              │   (Curation)    │
                └───────────────┘              └─────────────────┘              └─────────────────┘
```

## 📦 Components

| Component | Description | Port |
|-----------|-------------|------|
| **Kafka Producer** | Scans directories for images and publishes to Kafka | - |
| **Kafka Consumer** | Consumes images, runs VLM inference, stores results | - |
| **PostgreSQL** | Stores inference results and curated data | 5432 |
| **MinIO** | S3-compatible object storage for images | 9000 (API), 9001 (Console) |
| **pgAdmin** | PostgreSQL web UI | 5050 |
| **Web App** | Curation interface for reviewing/correcting VLM results | 5000 |

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone/copy the project to your VM
cd /path/to/highlander

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

See detailed instructions below.

---

## 📋 Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL 14+
- Kafka cluster (external or local)

---

## 🔧 Detailed Installation

### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib
```

### 2. PostgreSQL Setup

```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
ALTER USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE highlander;
GRANT ALL PRIVILEGES ON DATABASE highlander TO postgres;
EOF

# Allow password authentication (edit pg_hba.conf)
sudo nano /etc/postgresql/*/main/pg_hba.conf
# Change: local all postgres peer
# To:     local all postgres md5
# Add for Docker: host all all 172.17.0.0/16 md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### 3. Create Database Tables

```bash
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
```

### 4. Python Environment

```bash
cd /path/to/highlander

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Start Docker Services (MinIO & pgAdmin)

```bash
# Start MinIO and pgAdmin
docker-compose up -d

# Verify containers are running
docker ps
```

### 6. Configure MinIO

```bash
# Run MinIO setup script
source venv/bin/activate
python setup_minio.py
```

Or manually via MinIO Console:
1. Open `http://<YOUR_IP>:9001`
2. Login: `minioadmin` / `minioadmin123`
3. Create bucket: `highlander-crops`
4. Set bucket policy to public read

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or export these variables:

```bash
# Kafka
export KAFKA_BOOTSTRAP_SERVERS="your-kafka-server:9092"
export KAFKA_TOPIC_INPUT="crop-images"
export KAFKA_TOPIC_OUTPUT="vlm-results"

# PostgreSQL
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="highlander"
export DB_USER="postgres"
export DB_PASSWORD="postgres"

# MinIO
export MINIO_ENDPOINT="<YOUR_PUBLIC_IP>:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin123"
export MINIO_BUCKET="highlander-crops"

# VLM API (OpenAI compatible)
export OPENAI_API_KEY="your-api-key"
export VLM_MODEL="gpt-4-vision-preview"
```

### Update Configuration Files

**Important:** Replace `<YOUR_PUBLIC_IP>` with your VM's public IP in:

1. `kafka_consumer.py` - Line with `MINIO_ENDPOINT`
2. `webapp/app.py` - Line with `MINIO_PUBLIC_URL`
3. `setup_minio.py` - Line with `MINIO_ENDPOINT`
4. `insert_dummy_data.py` - Line with `MINIO_ENDPOINT`

```bash
# Find and replace (example for IP 35.207.192.14)
sed -i 's/localhost:9000/35.207.192.14:9000/g' kafka_consumer.py
sed -i 's/localhost:9000/35.207.192.14:9000/g' webapp/app.py
sed -i 's/localhost:9000/35.207.192.14:9000/g' setup_minio.py
sed -i 's/localhost:9000/35.207.192.14:9000/g' insert_dummy_data.py
```

---

## 🏃 Running the System

### Start All Services

```bash
# 1. Start Docker services (MinIO, pgAdmin)
docker-compose up -d

# 2. Activate Python environment
source venv/bin/activate

# 3. Start Kafka Consumer (VLM inference)
python kafka_consumer.py &

# 4. Start Kafka Producer (image scanner)
python kafka_crop_producer.py &

# 5. Start Web App
cd webapp && python app.py &
```

### Using the Provided Script

```bash
./start_services.sh
```

---

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Web App (OLY VISION)** | `http://<YOUR_IP>:5000` | admin/admin123, curator/curator123 |
| **MinIO Console** | `http://<YOUR_IP>:9001` | minioadmin/minioadmin123 |
| **MinIO API** | `http://<YOUR_IP>:9000` | - |
| **pgAdmin** | `http://<YOUR_IP>:5050` | admin@admin.com/admin123 |
| **PostgreSQL** | `<YOUR_IP>:5432` | postgres/postgres |

---

## 🔥 Firewall Rules (GCP/Cloud)

Open these ports in your cloud firewall:

```bash
# GCP example
gcloud compute firewall-rules create highlander-services \
    --allow tcp:5000,tcp:5050,tcp:9000,tcp:9001,tcp:5432 \
    --source-ranges 0.0.0.0/0 \
    --description "OLY VISION services"
```

| Port | Service |
|------|---------|
| 5000 | Web App |
| 5050 | pgAdmin |
| 9000 | MinIO API |
| 9001 | MinIO Console |
| 5432 | PostgreSQL |

---

## 📁 Project Structure

```
highlander/
├── docker-compose.yml      # MinIO & pgAdmin containers
├── kafka_consumer.py       # VLM inference consumer
├── kafka_crop_producer.py  # Image producer
├── setup_minio.py          # MinIO bucket setup
├── insert_dummy_data.py    # Test data generator
├── requirements.txt        # Python dependencies
├── setup.sh                # Automated setup script
├── start_services.sh       # Service startup script
├── README.md               # This file
└── webapp/
    ├── app.py              # Flask application
    ├── requirements.txt    # Web app dependencies
    ├── static/             # CSS, JS, images
    └── templates/          # HTML templates
        ├── login.html
        ├── main.html
        └── curation.html
```

---

## 🧪 Testing

### Insert Dummy Data

```bash
source venv/bin/activate
python insert_dummy_data.py
```

### Verify Database

```bash
sudo -u postgres psql -d highlander -c "SELECT COUNT(*) FROM vlm_inference;"
sudo -u postgres psql -d highlander -c "SELECT * FROM vlm_inference LIMIT 5;"
```

### Check MinIO

```bash
# List buckets
docker exec highlander-minio mc ls local/

# List images in bucket
docker exec highlander-minio mc ls local/highlander-crops/
```

---

## 🔧 Troubleshooting

### Images not loading in Web App

1. Check MinIO is running: `docker ps | grep minio`
2. Verify bucket exists and has public policy
3. Ensure MinIO URL uses public IP (not localhost)
4. Check firewall allows port 9000

### Database connection failed

1. Check PostgreSQL is running: `sudo systemctl status postgresql`
2. Verify password authentication in `pg_hba.conf`
3. Restart PostgreSQL: `sudo systemctl restart postgresql`

### pgAdmin can't connect to PostgreSQL

1. Add Docker network to `pg_hba.conf`:
   ```
   host all all 172.17.0.0/16 md5
   host all all 172.18.0.0/16 md5
   ```
2. Use host `172.17.0.1` in pgAdmin server config
3. Restart PostgreSQL

### Kafka consumer not receiving messages

1. Check Kafka cluster is accessible
2. Verify topic exists
3. Check consumer group offset

---

## 📝 Database Schema

### vlm_inference

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| store_id | INTEGER | Store identifier |
| uuid | VARCHAR(36) | Unique image ID |
| image_url | TEXT | Local file path |
| minio_url | TEXT | MinIO URL |
| age_category | VARCHAR(20) | Predicted age group |
| gender_category | VARCHAR(20) | Predicted gender |
| is_staff | BOOLEAN | Is staff member |
| created_at | TIMESTAMP | Record creation time |

### vlm_curated

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| inference_id | INTEGER | FK to vlm_inference |
| store_id | INTEGER | Store identifier |
| uuid | VARCHAR(36) | Unique image ID |
| image_url | TEXT | Local file path |
| minio_url | TEXT | MinIO URL |
| age_category | VARCHAR(20) | Curated age group |
| gender_category | VARCHAR(20) | Curated gender |
| is_staff | BOOLEAN | Is staff member |
| curated_by | VARCHAR(100) | Curator username |
| curated_at | TIMESTAMP | Curation time |

---
## DEMO


https://github.com/user-attachments/assets/e6894b4e-1187-41c1-ad80-e8b4d60771d7




## 📄 License

Internal use only.

---

## 👥 Support

For issues or questions, contact the development team.

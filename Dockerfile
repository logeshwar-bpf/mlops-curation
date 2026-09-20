# OLY VISION (Highlander) - Curation & MLOps Suite Dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV, PostgreSQL, and network healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy all application code, templates, static files, and sample datasets
COPY . .

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    DB_HOST=postgres \
    DB_PORT=5432 \
    DB_NAME=highlander \
    DB_USER=postgres \
    DB_PASSWORD=postgres \
    MINIO_PUBLIC_URL=http://localhost:9000

EXPOSE 5000

# Run Flask application with automatic DB initialization and seed loading
CMD ["python", "webapp/app.py"]

#!/usr/bin/env python3
"""
insert_dummy_data.py
--------------------
Scans the crops directory, uploads images to MinIO, and inserts dummy VLM 
inference records into the vlm_inference table for testing purposes.

Usage:
    pip install psycopg2-binary minio
    python insert_dummy_data.py
"""

import os
import sys
import random
import re
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("ERROR: minio not installed. Run: pip install minio")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CROPS_DIR = "/home/nihit/dhanush/highlander/oly-deepstream/deepstream/data/line_crossing/crops"
STORE_ID = 18  # Default store ID for dummy data

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "highlander",
    "user":     "postgres",
    "password": "postgres",
}

MINIO_CONFIG = {
    "endpoint":   "localhost:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin123",
    "secure":     False,
    "bucket":     "highlander-crops",
}
MINIO_PUBLIC_URL = "http://35.207.192.14:9000"  # Public IP for external access

# Possible values for random generation
AGE_CATEGORIES = ["infant", "child", "teen", "adult", "mature"]
GENDER_CATEGORIES = ["male", "female"]

# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def get_minio_client() -> Minio:
    """Create MinIO client."""
    return Minio(
        MINIO_CONFIG["endpoint"],
        access_key=MINIO_CONFIG["access_key"],
        secret_key=MINIO_CONFIG["secret_key"],
        secure=MINIO_CONFIG["secure"]
    )


def init_minio():
    """Initialize MinIO bucket if it doesn't exist."""
    try:
        client = get_minio_client()
        bucket = MINIO_CONFIG["bucket"]
        
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"✓ Created MinIO bucket: {bucket}")
            
            # Set public read policy
            policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket}/*"]
                }]
            }
            client.set_bucket_policy(bucket, json.dumps(policy))
            print(f"✓ Set public read policy on bucket: {bucket}")
        else:
            print(f"✓ MinIO bucket exists: {bucket}")
        return True
    except Exception as e:
        print(f"⚠ MinIO not available: {e}")
        print("  Images will be stored with local paths only")
        return False


def upload_to_minio(image_path: str, store_id: int, uuid: str, event_type: str) -> str | None:
    """
    Upload image to MinIO and return the public URL.
    """
    try:
        client = get_minio_client()
        
        # Get current date for path organization
        now = datetime.now()
        year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
        
        # Build object path: store_18/2026/02/20/uuid_entry.jpg
        object_name = f"store_{store_id}/{year}/{month}/{day}/{uuid}_{event_type}.jpg"
        
        # Upload to MinIO
        client.fput_object(
            bucket_name=MINIO_CONFIG["bucket"],
            object_name=object_name,
            file_path=image_path,
            content_type="image/jpeg"
        )
        
        # Build public URL
        minio_url = f"{MINIO_PUBLIC_URL}/{MINIO_CONFIG['bucket']}/{object_name}"
        return minio_url
        
    except Exception as e:
        print(f"  ⚠ Failed to upload {uuid}: {e}")
        return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    """Create a new database connection."""
    return psycopg2.connect(**DB_CONFIG)


def init_database():
    """Create the vlm_inference table if it doesn't exist."""
    create_table_sql = """
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
    
    -- Add minio_url column if it doesn't exist (for existing tables)
    DO $$ 
    BEGIN 
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name='vlm_inference' AND column_name='minio_url') THEN
            ALTER TABLE vlm_inference ADD COLUMN minio_url TEXT;
        END IF;
    END $$;
    """
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    conn.close()
    print("✓ Database table 'vlm_inference' initialized")


def clear_existing_data():
    """Clear existing data from tables."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM vlm_curated")
        cur.execute("DELETE FROM vlm_inference")
    conn.commit()
    conn.close()
    print("✓ Cleared existing data from tables")


def parse_image_filename(filename: str) -> dict | None:
    """
    Extract UUID and event_type from filename like:
        f47ac10b-58cc-4372-a567-0e02b2c3d479_entry_20260219_091053_123.jpg
    """
    pattern = r"^(?P<uuid>[0-9a-f\-]{36})_(?P<event_type>entry|exit)_.*\.jpe?g$"
    m = re.match(pattern, filename, re.IGNORECASE)
    if not m:
        return None
    return {
        "uuid": m.group("uuid"),
        "event_type": m.group("event_type"),
    }


def scan_images(crops_dir: str, limit: int = None) -> list[dict]:
    """
    Scan the crops directory for images.
    Returns list of dicts with uuid, event_type, and image_path.
    """
    images = []
    crops_path = Path(crops_dir)

    if not crops_path.exists():
        print(f"ERROR: Crops directory does not exist: {crops_dir}")
        return images

    for person_dir in crops_path.iterdir():
        if not person_dir.is_dir():
            continue

        for event_type in ("entry", "exit"):
            event_dir = person_dir / event_type
            if not event_dir.is_dir():
                continue

            for img_file in event_dir.glob("*.jpg"):
                meta = parse_image_filename(img_file.name)
                if meta is None:
                    continue

                images.append({
                    "uuid": meta["uuid"],
                    "event_type": meta["event_type"],
                    "image_path": str(img_file.resolve()),
                })

                if limit and len(images) >= limit:
                    return images

    return images


def insert_records(records: list[tuple]):
    """Insert records into the database."""
    insert_sql = """
    INSERT INTO vlm_inference (store_id, uuid, image_url, minio_url, age_category, gender_category, is_staff)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        execute_batch(cur, insert_sql, records)
    conn.commit()
    conn.close()


def main():
    print("=" * 60)
    print("Dummy Data Inserter for vlm_inference table")
    print("=" * 60)
    print(f"  Crops dir : {CROPS_DIR}")
    print(f"  Database  : {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"  MinIO     : {MINIO_CONFIG['endpoint']}/{MINIO_CONFIG['bucket']}")
    print("=" * 60)

    # Initialize database
    init_database()
    
    # Clear existing data
    clear_existing_data()
    
    # Initialize MinIO
    minio_available = init_minio()

    # Scan for images
    print("\nScanning for images...")
    images = scan_images(CROPS_DIR)
    print(f"✓ Found {len(images)} image(s)")

    if not images:
        print("No images found. Exiting.")
        return

    # Generate records and upload to MinIO
    print("\nProcessing images...")
    records = []
    uploaded_count = 0
    
    for i, img in enumerate(images):
        # Upload to MinIO if available
        minio_url = None
        if minio_available:
            minio_url = upload_to_minio(
                img["image_path"],
                STORE_ID,
                img["uuid"],
                img["event_type"]
            )
            if minio_url:
                uploaded_count += 1
        
        # Generate random VLM predictions
        record = (
            STORE_ID,
            img["uuid"],
            img["image_path"],
            minio_url,
            random.choice(AGE_CATEGORIES),
            random.choice(GENDER_CATEGORIES),
            random.random() < 0.1,  # 10% chance of being staff
        )
        records.append(record)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")

    # Insert into database
    print(f"\nInserting {len(records)} record(s) into database...")
    insert_records(records)
    print(f"✓ Successfully inserted {len(records)} record(s)")
    
    if minio_available:
        print(f"✓ Uploaded {uploaded_count}/{len(images)} image(s) to MinIO")

    # Show sample of inserted data
    print("\n" + "=" * 60)
    print("Sample of inserted records:")
    print("=" * 60)
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, store_id, uuid, age_category, gender_category, is_staff,
                   CASE WHEN minio_url IS NOT NULL THEN 'Yes' ELSE 'No' END as has_minio
            FROM vlm_inference 
            ORDER BY id DESC 
            LIMIT 5
        """)
        rows = cur.fetchall()
        
        print(f"{'ID':<6} {'Store':<6} {'UUID':<38} {'Age':<8} {'Gender':<8} {'Staff':<6} {'MinIO'}")
        print("-" * 110)
        for row in rows:
            print(f"{row[0]:<6} {row[1] or 'N/A':<6} {row[2]:<38} {row[3]:<8} {row[4]:<8} {str(row[5]):<6} {row[6]}")
    
    conn.close()

    # Show total count
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM vlm_inference")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM vlm_inference WHERE minio_url IS NOT NULL")
        with_minio = cur.fetchone()[0]
    conn.close()
    
    print(f"\nTotal records in vlm_inference table: {total}")
    print(f"Records with MinIO URLs: {with_minio}")


if __name__ == "__main__":
    main()

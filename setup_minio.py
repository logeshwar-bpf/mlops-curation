#!/usr/bin/env python3
"""
setup_minio.py
--------------
Initialize MinIO bucket and configuration for Highlander.

Usage:
    pip install minio
    python setup_minio.py
"""

import sys

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("ERROR: minio not installed. Run: pip install minio")
    sys.exit(1)

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
MINIO_SECURE = False  # Set to True if using HTTPS

BUCKET_NAME = "highlander-crops"


def setup_minio():
    """Initialize MinIO bucket."""
    print("=" * 60)
    print("MinIO Setup for Highlander")
    print("=" * 60)
    print(f"  Endpoint: {MINIO_ENDPOINT}")
    print(f"  Bucket:   {BUCKET_NAME}")
    print("=" * 60)

    # Create client
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

    # Check if bucket exists
    try:
        if client.bucket_exists(BUCKET_NAME):
            print(f"✓ Bucket '{BUCKET_NAME}' already exists")
        else:
            # Create bucket
            client.make_bucket(BUCKET_NAME)
            print(f"✓ Created bucket '{BUCKET_NAME}'")

        # Set bucket policy for public read access
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{BUCKET_NAME}/*"]
                }
            ]
        }
        
        import json
        client.set_bucket_policy(BUCKET_NAME, json.dumps(policy))
        print(f"✓ Set public read policy on '{BUCKET_NAME}'")

        print("\n" + "=" * 60)
        print("MinIO setup complete!")
        print("=" * 60)
        print(f"  Console URL: http://localhost:9001")
        print(f"  API URL:     http://localhost:9000")
        print(f"  Bucket URL:  http://localhost:9000/{BUCKET_NAME}/")
        print("=" * 60)

    except S3Error as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_minio()


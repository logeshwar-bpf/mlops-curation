#!/usr/bin/env python3
"""
falcon_gender_consumer.py
--------------------------
Consumes crop images from 'falcon-gender' Kafka topic in batches of 10,
runs concurrent VLM predictions (staff, gender, age group),
uploads images to MinIO, stores results in PostgreSQL, 
and publishes results to 'falcon-gender-push'.

Input message (from kafka_crop_producer.py):
{
    "store_id": 18,
    "uuid": "f47ac10b-...",
    "event_type": "entry" | "exit",
    "timestamp": "2026-02-19T09:10:53.123Z",
    "image_b64": "<base64 JPEG>",
    "image_path": "/absolute/path/to/image.jpg"
}

Output message (to falcon-gender-push):
{
    "store_id": 18,
    "uuid": "f47ac10b-...",
    "event_type": "entry" | "exit",
    "timestamp": "2026-02-19T09:10:53.123Z",
    "is_staff": true | false,
    "gender": "male" | "female" | "unknown",
    "age_group": "infant" | "child" | "teen" | "adult" | "mature" | "unknown",
    "minio_url": "http://localhost:9000/highlander-crops/..."
}

Database table (vlm_inference):
    - id (SERIAL PRIMARY KEY)
    - store_id (INTEGER)
    - uuid (VARCHAR)
    - image_url (TEXT) - local path to image
    - minio_url (TEXT) - MinIO URL for image
    - age_category (VARCHAR)
    - gender_category (VARCHAR)
    - is_staff (BOOLEAN)
    - created_at (TIMESTAMP)

Usage:
    pip install confluent-kafka openai opencv-python psycopg2-binary minio
    python falcon_gender_consumer.py
"""

import sys
import json
import base64
import asyncio
import logging
import time

import cv2
import numpy as np

try:
    from confluent_kafka import Consumer, Producer, KafkaError, TopicPartition
except ImportError:
    print("ERROR: confluent-kafka not installed. Run: pip install confluent-kafka")
    sys.exit(1)

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from minio import Minio
    from io import BytesIO
except ImportError:
    print("ERROR: minio not installed. Run: pip install minio")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_BROKERS     = "kafka.poc.oly.live:9094"
CONSUMER_TOPIC    = "falcon-gender-best"
PRODUCER_TOPIC    = "falcon-gender-push-2"
CONSUMER_GROUP    = "falcon-gender-consumer-group"

BATCH_SIZE        = 10    # messages pulled and inferred together
POLL_TIMEOUT      = 1.0   # seconds to wait for each poll
BATCH_TIMEOUT     = 5.0   # max seconds to wait to fill a batch before processing anyway

LLAMA_URL         = "http://35.207.192.14:8081/v1"
MODEL_NAME        = "llava"

KAFKA_SASL = {
    "security.protocol": "SASL_PLAINTEXT",
    "sasl.mechanism":    "SCRAM-SHA-256",
    "sasl.username":     "kafka-admin",
    "sasl.password":     "kafka-admin-password",
}

# Database config
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "highlander",
    "user":     "postgres",
    "password": "postgres",
}

# MinIO config
MINIO_CONFIG = {
    "endpoint":   "localhost:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin123",
    "secure":     False,
    "bucket":     "highlander-crops",
}
MINIO_PUBLIC_URL = "http://35.207.192.14:9000"  # Public URL for accessing images

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("falcon_gender_consumer")

# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------
vlm_client = AsyncOpenAI(base_url=LLAMA_URL, api_key="sk-no-key-required")

SYSTEM_PROMPT = """You are a vision AI analyzing a CCTV person crop image.
You must respond with ONLY a JSON object. No explanation. No markdown. No extra text.

Example response:
{"is_staff": false, "gender": "male", "age_group": "adult"}

Fields:
- is_staff (boolean): true ONLY if the person is wearing a black security uniform, else false.
- gender (string): "male", "female", or "unknown". Use body shape, face, hair — not clothing.
- age_group (string): visually estimate the person's age and pick the closest bucket:
    "infant"  — 0 to 2 years old
    "child"   — 3 to 12 years old
    "teen"    — 13 to 21 years old
    "adult"   — 22 to 35 years old
    "mature"  — older than 35 years old
    "unknown" — only if the person is completely not visible

Always include all three fields. Never omit age_group. Make your best estimate even if unsure.
"""

USER_PROMPT = 'Analyze this person image. Reply with JSON only: {"is_staff": ..., "gender": ..., "age_group": ...}'


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_and_resize(image_b64: str) -> str | None:
    """Decode base64 JPEG, resize to 224x448, re-encode to base64."""
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (224, 448))
        _, buf = cv2.imencode(".jpg", img)
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error("Image decode/resize failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# VLM prediction (single image)
# ---------------------------------------------------------------------------

async def predict_one(item: dict) -> dict:
    """
    Run VLM prediction for a single message dict.
    Returns the output message dict (no image_b64).
    """
    uuid       = item.get("uuid", "unknown")
    image_b64  = item.get("image_b64")

    if not image_b64:
        log.warning("uuid=%s — missing image_b64, skipping", uuid)
        return _build_output(item, is_staff=False, gender="unknown", age_group="unknown")

    resized_b64 = decode_and_resize(image_b64)
    if resized_b64 is None:
        log.warning("uuid=%s — image decode failed", uuid)
        return _build_output(item, is_staff=False, gender="unknown", age_group="unknown")

    try:
        completion = await vlm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{resized_b64}"}}
                ]}
            ],
            temperature=0.0,
            max_tokens=120,
        )

        raw = completion.choices[0].message.content.strip()
        predictions = _parse_response(raw)
        log.info("uuid=%s → is_staff=%s gender=%s age=%s",
                 uuid, predictions["is_staff"], predictions["gender"], predictions["age_group"])
        return _build_output(item, **predictions)

    except Exception as e:
        log.error("uuid=%s — VLM call failed: %s", uuid, e)
        return _build_output(item, is_staff=False, gender="unknown", age_group="unknown")


def _build_output(item: dict, is_staff: bool, gender: str, age_group: str) -> dict:
    return {
        "store_id":   item.get("store_id"),
        "uuid":       item.get("uuid"),
        "event_type": item.get("event_type"),
        "timestamp":  item.get("timestamp"),
        "is_staff":   is_staff,
        "gender":     gender,
        "age_group":  age_group,
    }


def _parse_response(raw: str) -> dict:
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        return {
            "is_staff":  bool(data.get("is_staff", False)),
            "gender":    _validate(data.get("gender", "unknown"), ("male", "female", "unknown")),
            "age_group": _validate(data.get("age_group", "unknown"),
                                   ("infant", "child", "teen", "adult", "mature", "unknown")),
        }
    except json.JSONDecodeError:
        log.warning("VLM returned non-JSON, using keyword fallback. Raw: %s", raw)
        return _keyword_fallback(raw)


def _keyword_fallback(text: str) -> dict:
    t = text.lower()
    is_staff  = "true" in t and "is_staff" in t
    gender    = "female" if "female" in t else ("male" if "male" in t else "unknown")
    age_group = "unknown"
    for ag in ("infant", "child", "teen", "adult", "mature"):
        if ag in t:
            age_group = ag
            break
    return {"is_staff": is_staff, "gender": gender, "age_group": age_group}


def _validate(val: str, allowed: tuple) -> str:
    return val if val in allowed else "unknown"


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

async def predict_batch(items: list[dict]) -> list[dict]:
    """Run VLM predictions on a batch concurrently."""
    tasks = [predict_one(item) for item in items]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------

def make_consumer() -> Consumer:
    cfg = {
        "bootstrap.servers":  KAFKA_BROKERS,
        "group.id":           CONSUMER_GROUP,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,   # manual commit after publish
        **KAFKA_SASL,
    }
    c = Consumer(cfg)
    c.subscribe([CONSUMER_TOPIC])
    log.info("Consumer subscribed to: %s", CONSUMER_TOPIC)
    return c


def make_producer() -> Producer:
    cfg = {
        "bootstrap.servers":  KAFKA_BROKERS,
        "client.id":          "falcon-gender-push-producer",
        "acks":               "all",
        "retries":            3,
        "batch.size":         16384,
        "linger.ms":          10,
        "request.timeout.ms": 30000,
        "retry.backoff.ms":   100,
        "message.max.bytes":  1_000_000,
        **KAFKA_SASL,
    }
    return Producer(cfg)


def delivery_report(err, msg):
    if err:
        log.error("Delivery failed [%s]: %s", msg.key(), err)
    else:
        log.debug("Published → %s [partition %d] offset %d",
                  msg.topic(), msg.partition(), msg.offset())


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
            log.info("Created MinIO bucket: %s", bucket)
            
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
            log.info("Set public read policy on bucket: %s", bucket)
        else:
            log.info("MinIO bucket exists: %s", bucket)
    except Exception as e:
        log.error("Failed to initialize MinIO: %s", e)
        raise


def upload_to_minio(image_b64: str, store_id: int, uuid: str, event_type: str, timestamp: str) -> str | None:
    """
    Upload image to MinIO and return the public URL.
    
    Args:
        image_b64: Base64 encoded image
        store_id: Store ID for path organization
        uuid: Unique identifier
        event_type: 'entry' or 'exit'
        timestamp: ISO timestamp for date organization
    
    Returns:
        Public URL of uploaded image, or None on failure
    """
    try:
        # Decode image
        image_bytes = base64.b64decode(image_b64)
        
        # Parse timestamp for path organization (e.g., 2026-02-19T09:10:53.123Z)
        date_part = timestamp[:10] if timestamp else "unknown"
        year, month, day = date_part.split("-") if "-" in date_part else ("unknown", "unknown", "unknown")
        
        # Build object path: store_18/2026/02/19/uuid_entry.jpg
        object_name = f"store_{store_id}/{year}/{month}/{day}/{uuid}_{event_type}.jpg"
        
        # Upload to MinIO
        client = get_minio_client()
        client.put_object(
            bucket_name=MINIO_CONFIG["bucket"],
            object_name=object_name,
            data=BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/jpeg"
        )
        
        # Build public URL
        minio_url = f"{MINIO_PUBLIC_URL}/{MINIO_CONFIG['bucket']}/{object_name}"
        log.debug("Uploaded to MinIO: %s", minio_url)
        return minio_url
        
    except Exception as e:
        log.error("Failed to upload to MinIO: %s", e)
        return None


def upload_batch_to_minio(batch_items: list[dict]) -> list[str]:
    """
    Upload a batch of images to MinIO.
    
    Returns:
        List of MinIO URLs (empty string if upload failed)
    """
    minio_urls = []
    for item in batch_items:
        image_b64 = item.get("image_b64")
        if not image_b64:
            minio_urls.append("")
            continue
            
        url = upload_to_minio(
            image_b64=image_b64,
            store_id=item.get("store_id", 0),
            uuid=item.get("uuid", "unknown"),
            event_type=item.get("event_type", "unknown"),
            timestamp=item.get("timestamp", "")
        )
        minio_urls.append(url or "")
    
    return minio_urls


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
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        conn.close()
        log.info("Database table 'vlm_inference' initialized")
    except Exception as e:
        log.error("Failed to initialize database: %s", e)
        raise


def save_to_database(results: list[dict], image_paths: list[str], minio_urls: list[str]):
    """
    Save VLM inference results to the database.
    
    Args:
        results: List of result dicts from VLM prediction
        image_paths: List of corresponding local image paths
        minio_urls: List of corresponding MinIO URLs
    """
    insert_sql = """
    INSERT INTO vlm_inference (store_id, uuid, image_url, minio_url, age_category, gender_category, is_staff)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    records = []
    for result, image_path, minio_url in zip(results, image_paths, minio_urls):
        records.append((
            result.get("store_id"),
            result.get("uuid"),
            image_path,
            minio_url,
            result.get("age_group"),
            result.get("gender"),
            result.get("is_staff"),
        ))
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_batch(cur, insert_sql, records)
        conn.commit()
        conn.close()
        log.info("Saved %d record(s) to database", len(records))
    except Exception as e:
        log.error("Failed to save to database: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    log.info("=" * 60)
    log.info("Falcon Gender Consumer (batch=%d)", BATCH_SIZE)
    log.info("  Consuming from : %s", CONSUMER_TOPIC)
    log.info("  Publishing to  : %s", PRODUCER_TOPIC)
    log.info("  VLM endpoint   : %s", LLAMA_URL)
    log.info("  Database       : %s@%s:%s/%s", 
             DB_CONFIG["user"], DB_CONFIG["host"], DB_CONFIG["port"], DB_CONFIG["database"])
    log.info("  MinIO          : %s/%s", MINIO_CONFIG["endpoint"], MINIO_CONFIG["bucket"])
    log.info("=" * 60)

    # Initialize database table
    init_database()
    
    # Initialize MinIO bucket
    init_minio()

    consumer = make_consumer()
    producer = make_producer()

    try:
        while True:
            # ----------------------------------------------------------------
            # 1. Collect a batch of up to BATCH_SIZE messages
            # ----------------------------------------------------------------
            batch_msgs  = []   # raw confluent_kafka Message objects (for commit)
            batch_items = []   # parsed dicts (for VLM)
            batch_start = time.time()

            while len(batch_msgs) < BATCH_SIZE:
                # Stop filling if we've waited too long (process partial batch)
                if time.time() - batch_start > BATCH_TIMEOUT and batch_msgs:
                    break

                msg = consumer.poll(timeout=POLL_TIMEOUT)

                if msg is None:
                    if batch_msgs:
                        break   # nothing new — process what we have
                    continue    # empty queue, keep waiting

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        if batch_msgs:
                            break
                    else:
                        log.error("Consumer error: %s", msg.error())
                    continue

                try:
                    item = json.loads(msg.value())
                    batch_msgs.append(msg)
                    batch_items.append(item)
                except json.JSONDecodeError as e:
                    log.error("Invalid JSON in message: %s", e)

            if not batch_msgs:
                continue

            # ----------------------------------------------------------------
            # 2. Run VLM predictions concurrently on the batch
            # ----------------------------------------------------------------
            log.info("Running batch inference on %d image(s)...", len(batch_items))
            t0 = time.time()
            results = await predict_batch(batch_items)
            elapsed = time.time() - t0
            log.info("Batch of %d done in %.2fs (%.2fs/img)",
                     len(results), elapsed, elapsed / len(results))

            # ----------------------------------------------------------------
            # 3. Upload images to MinIO
            # ----------------------------------------------------------------
            log.info("Uploading %d image(s) to MinIO...", len(batch_items))
            minio_urls = upload_batch_to_minio(batch_items)
            uploaded_count = sum(1 for url in minio_urls if url)
            log.info("Uploaded %d/%d image(s) to MinIO", uploaded_count, len(batch_items))

            # ----------------------------------------------------------------
            # 4. Save results to database
            # ----------------------------------------------------------------
            image_paths = [item.get("image_path", "") for item in batch_items]
            save_to_database(results, image_paths, minio_urls)

            # ----------------------------------------------------------------
            # 5. Publish all results, then commit offsets
            # ----------------------------------------------------------------
            for result, minio_url in zip(results, minio_urls):
                # Add minio_url to the result for downstream consumers
                result_with_url = {**result, "minio_url": minio_url}
                producer.produce(
                    topic=PRODUCER_TOPIC,
                    key=(result.get("uuid") or "unknown").encode("utf-8"),
                    value=json.dumps(result_with_url).encode("utf-8"),
                    callback=delivery_report,
                )
                producer.poll(0)

            producer.flush(timeout=15)

            # Commit the highest offset per partition in this batch
            consumer.commit(asynchronous=False)
            log.info("Committed offsets for batch of %d", len(batch_msgs))

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        producer.flush(timeout=30)
        consumer.close()
        log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())

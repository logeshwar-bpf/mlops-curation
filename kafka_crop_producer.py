#!/usr/bin/env python3
"""
kafka_crop_producer.py
----------------------
Polls the line_crossing crops directory for new entry/exit images,
encodes them as base64, and publishes them to a Kafka topic.

Kafka message schema:
{
    "store_id": 18,
    "camera_id": 97,
    "uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "event_type": "entry" | "exit",
    "image_b64": "<base64-encoded JPEG>",
    "timestamp": "2026-02-19T09:10:53.123Z",
    "image_path": "/absolute/path/to/image.jpg"
}

Usage:
    pip install confluent-kafka

    python kafka_crop_producer.py \
        --crops-dir /home/nihit/dhanush/highlander/oly-deepstream/deepstream/data/line_crossing/crops \
        --sources-json /home/nihit/dhanush/highlander/oly-deepstream/deepstream/configs/pipelines/retail_reid/sources.json \
        --kafka-brokers localhost:9092 \
        --kafka-topic retail-reid-crops \
        --poll-interval 5 \
        --state-file /tmp/kafka_crop_sent.json
"""

import os
import sys
import json
import base64
import time
import logging
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from confluent_kafka import Producer
except ImportError:
    print("ERROR: confluent-kafka not installed. Run: pip install confluent-kafka")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("kafka_crop_producer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sources(sources_json_path: str) -> dict:
    """
    Load sources.json and return a dict keyed by camera_id.
    Each value is the full source dict (store_id, camera_id, flow, ...).
    """
    with open(sources_json_path) as f:
        data = json.load(f)
    return {s["camera_id"]: s for s in data.get("sources", [])}


def load_state(state_file: str) -> set:
    """Load the set of already-sent image paths from the state file."""
    if os.path.exists(state_file):
        with open(state_file) as f:
            return set(json.load(f))
    return set()


def save_state(state_file: str, sent: set):
    """Persist the set of sent image paths to disk."""
    with open(state_file, "w") as f:
        json.dump(list(sent), f)


def parse_image_filename(filename: str) -> dict | None:
    """
    Extract metadata from filename like:
        f47ac10b-58cc-4372-a567-0e02b2c3d479_entry_20260219_091053_123.jpg

    Returns dict with uuid, event_type, timestamp, or None if no match.
    """
    pattern = (
        r"^(?P<uuid>[0-9a-f\-]{36})"
        r"_(?P<event_type>entry|exit)"
        r"_(?P<date>\d{8})"
        r"_(?P<time>\d{6})"
        r"_(?P<ms>\d+)"
        r"\.jpe?g$"
    )
    m = re.match(pattern, filename, re.IGNORECASE)
    if not m:
        return None

    d, t, ms = m.group("date"), m.group("time"), m.group("ms")
    # Parse to ISO-8601 UTC
    try:
        dt_naive = datetime.strptime(f"{d}{t}", "%Y%m%d%H%M%S")
        iso_ts = dt_naive.replace(tzinfo=timezone.utc).strftime(
            f"%Y-%m-%dT%H:%M:%S.{ms.zfill(3)}Z"
        )
    except ValueError:
        iso_ts = datetime.now(timezone.utc).isoformat()

    return {
        "uuid": m.group("uuid"),
        "event_type": m.group("event_type"),
        "timestamp": iso_ts,
    }


def scan_new_images(crops_dir: str, sent: set) -> list[dict]:
    """
    Walk the crops directory for new .jpg images not yet in sent.

    Expected structure:
        crops/
          person_<uuid>/
            entry/
              <uuid>_entry_<date>_<time>_<ms>.jpg
            exit/
              <uuid>_exit_<date>_<time>_<ms>.jpg
    """
    new_images = []
    crops_path = Path(crops_dir)

    if not crops_path.exists():
        log.warning("Crops directory does not exist: %s", crops_dir)
        return new_images

    for person_dir in crops_path.iterdir():
        if not person_dir.is_dir():
            continue

        for event_type in ("entry", "exit"):
            event_dir = person_dir / event_type
            if not event_dir.is_dir():
                continue

            for img_file in event_dir.glob("*.jpg"):
                abs_path = str(img_file.resolve())
                if abs_path in sent:
                    continue  # already sent

                meta = parse_image_filename(img_file.name)
                if meta is None:
                    log.warning("Skipping unrecognised filename: %s", img_file.name)
                    continue

                new_images.append({
                    "abs_path": abs_path,
                    "meta": meta,
                })

    return new_images


def build_message(img_info: dict, sources: dict) -> dict:
    """
    Build the Kafka message payload for an image.
    store_id / camera_id are taken from sources.json (first source entry).
    """
    meta = img_info["meta"]
    abs_path = img_info["abs_path"]

    # Read & encode image
    with open(abs_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Use first source for store_id
    first_source = next(iter(sources.values())) if sources else {}

    return {
        "store_id": first_source.get("store_id"),
        "uuid": meta["uuid"],
        "event_type": meta["event_type"],
        "timestamp": meta["timestamp"],
        "image_b64": image_b64,
        "image_path": abs_path,
    }


def delivery_report(err, msg):
    """Called once for each produced message."""
    if err:
        log.error("Delivery failed for %s: %s", msg.key(), err)
    else:
        log.info(
            "Delivered to %s [partition %d] offset %d",
            msg.topic(), msg.partition(), msg.offset()
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    log.info("Starting kafka_crop_producer")
    log.info("  Crops dir    : %s", args.crops_dir)
    log.info("  Sources JSON : %s", args.sources_json)
    log.info("  Kafka broker : %s", args.kafka_brokers)
    log.info("  Kafka topic  : %s", args.kafka_topic)
    log.info("  Poll interval: %ds", args.poll_interval)
    log.info("  State file   : %s", args.state_file)

    sources = load_sources(args.sources_json)
    log.info("Loaded %d source(s) from sources.json", len(sources))

    producer = Producer({
        "bootstrap.servers": args.kafka_brokers,
        "client.id": "kafka_crop_producer",
        "acks": "all",
        "retries": 3,
        "batch.size": 16384,
        "linger.ms": 10,

        "message.max.bytes": 10_000_000,   # bumped to handle base64 images
        "request.timeout.ms": 30000,
        "retry.backoff.ms": 100,
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-256",
        "sasl.username": "kafka-admin",
        "sasl.password": "kafka-admin-password",
    })

    sent = load_state(args.state_file)
    log.info("Resuming with %d already-sent images in state", len(sent))

    try:
        while True:
            new_images = scan_new_images(args.crops_dir, sent)

            if new_images:
                log.info("Found %d new image(s) to send", len(new_images))
            else:
                log.debug("No new images found")

            for img_info in new_images:
                try:
                    payload = build_message(img_info, sources)
                    key = f"{payload['uuid']}_{payload['event_type']}"

                    producer.produce(
                        topic=args.kafka_topic,
                        key=key.encode("utf-8"),
                        value=json.dumps(payload).encode("utf-8"),
                        callback=delivery_report,
                    )
                    producer.poll(0)  # Trigger delivery callbacks without blocking

                    sent.add(img_info["abs_path"])
                    save_state(args.state_file, sent)

                except FileNotFoundError:
                    log.error("Image file not found (deleted?): %s", img_info["abs_path"])
                except Exception as e:
                    log.error("Failed to produce message for %s: %s", img_info["abs_path"], e)

            # Flush pending messages before sleeping
            if new_images:
                producer.flush(timeout=10)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        log.info("Shutting down — flushing remaining messages...")
        producer.flush(timeout=30)
        log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Poll crops directory and publish new images to Kafka."
    )
    parser.add_argument(
        "--crops-dir",
        default="/home/nihit/dhanush/highlander/oly-deepstream/deepstream/data/line_crossing/crops",
        help="Root crops directory to watch",
    )
    parser.add_argument(
        "--sources-json",
        default="/home/nihit/dhanush/highlander/oly-deepstream/deepstream/configs/pipelines/retail_reid/sources.json",
        help="Path to sources.json",
    )
    parser.add_argument(
        "--kafka-brokers",
        default="kafka.poc.oly.live:9094",
        help="Kafka bootstrap servers (default: kafka.poc.oly.live:9094)",
    )
    parser.add_argument(
        "--kafka-topic",
        default="falcon-gender",
        help="Kafka topic to publish to",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between directory scans (default: 5)",
    )
    parser.add_argument(
        "--state-file",
        default="/tmp/kafka_crop_sent.json",
        help="JSON file to persist sent-image state across restarts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
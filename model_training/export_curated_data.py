#!/usr/bin/env python3
"""
OLY VISION - Export Curated Data for Model Training
----------------------------------------------------
Exports curated images from the database to folder structure for training.
Excludes 'unknown' labels and supports stratified splitting.

Features:
- Export by task (gender, age, staff)
- Exclude unknown labels
- Download images from MinIO
- Generate metadata.json with statistics
- Stratified train/valid/test split

Usage:
    python export_curated_data.py --task gender --output /data/exports
    python export_curated_data.py --task age --output /data/exports --split 70 15 15
    python export_curated_data.py --task staff --output /data/exports
"""

import os
import sys
import json
import argparse
import hashlib
import requests
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

from tqdm import tqdm

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "highlander",
    "user": "postgres",
    "password": "postgres",
}

# Task configurations - classes to include (excluding unknown)
TASK_CONFIG = {
    "gender": {
        "column": "gender_category",
        "classes": ["male", "female"],
        "exclude": ["unknown", None, ""],
    },
    "age": {
        "column": "age_category",
        "classes": ["infant", "child", "teen", "adult", "mature"],
        "exclude": ["unknown", None, ""],
    },
    "staff": {
        "column": "is_staff",
        "classes": ["staff", "non_staff"],
        "exclude": [None],
        "is_boolean": True,
    },
}


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)


def fetch_curated_data(task: str) -> list:
    """
    Fetch curated data from database for a specific task.
    Excludes unknown labels.
    """
    config = TASK_CONFIG[task]
    column = config["column"]
    exclude = config["exclude"]
    is_boolean = config.get("is_boolean", False)
    
    conn = get_db_connection()
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if is_boolean:
            # For staff (boolean column)
            cur.execute("""
                SELECT 
                    vc.id,
                    vc.inference_id,
                    vc.store_id,
                    vc.uuid,
                    vc.minio_url,
                    vi.minio_url as inference_minio_url,
                    vc.is_staff
                FROM vlm_curated vc
                LEFT JOIN vlm_inference vi ON vc.inference_id = vi.id
                WHERE vc.is_staff IS NOT NULL
            """)
        else:
            # For gender/age (string column)
            placeholders = ','.join(['%s'] * len(config["classes"]))
            cur.execute(f"""
                SELECT 
                    vc.id,
                    vc.inference_id,
                    vc.store_id,
                    vc.uuid,
                    vc.minio_url,
                    vi.minio_url as inference_minio_url,
                    vc.{column} as label
                FROM vlm_curated vc
                LEFT JOIN vlm_inference vi ON vc.inference_id = vi.id
                WHERE vc.{column} IN ({placeholders})
                  AND vc.{column} IS NOT NULL
                  AND vc.{column} != 'unknown'
                  AND vc.{column} != ''
            """, config["classes"])
        
        data = cur.fetchall()
    
    conn.close()
    
    # Process data
    processed = []
    for row in data:
        # Get the best available URL
        minio_url = row.get('minio_url') or row.get('inference_minio_url')
        
        if not minio_url:
            continue
        
        # Determine label
        if is_boolean:
            label = "staff" if row['is_staff'] else "non_staff"
        else:
            label = row['label'].lower().strip()
        
        # Skip if label not in allowed classes
        if label not in config["classes"]:
            continue
        
        processed.append({
            'id': row['id'],
            'store_id': row['store_id'],
            'uuid': row['uuid'],
            'minio_url': minio_url,
            'label': label,
        })
    
    return processed


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download image from URL to local path."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def stratified_split(data: list, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int = 42) -> dict:
    """
    Perform stratified split maintaining class proportions.
    
    Args:
        data: List of data items with 'label' field
        train_ratio: Fraction for training (e.g., 0.7)
        valid_ratio: Fraction for validation (e.g., 0.15)
        test_ratio: Fraction for test (e.g., 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'valid', 'test' lists
    """
    import random
    random.seed(seed)
    
    # Group by label
    by_label = defaultdict(list)
    for item in data:
        by_label[item['label']].append(item)
    
    splits = {'train': [], 'valid': [], 'test': []}
    
    for label, items in by_label.items():
        # Shuffle items
        random.shuffle(items)
        
        n = len(items)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        
        splits['train'].extend(items[:n_train])
        splits['valid'].extend(items[n_train:n_train + n_valid])
        splits['test'].extend(items[n_train + n_valid:])
    
    # Shuffle each split
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    return splits


def export_data(task: str, output_dir: str, split_ratios: tuple = (0.7, 0.15, 0.15), seed: int = 42):
    """
    Main export function.
    
    Args:
        task: 'gender', 'age', or 'staff'
        output_dir: Base output directory
        split_ratios: (train, valid, test) ratios
        seed: Random seed
    """
    if task not in TASK_CONFIG:
        print(f"ERROR: Unknown task '{task}'. Available: {list(TASK_CONFIG.keys())}")
        sys.exit(1)
    
    config = TASK_CONFIG[task]
    train_ratio, valid_ratio, test_ratio = split_ratios
    
    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.01:
        print(f"ERROR: Split ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  OLY VISION - Data Export")
    print(f"  Task: {task.upper()}")
    print(f"  Classes: {config['classes']}")
    print(f"  Excluding: unknown labels")
    print(f"{'='*60}\n")
    
    # Fetch data from database
    print("[1/4] Fetching curated data from database...")
    data = fetch_curated_data(task)
    
    if not data:
        print("ERROR: No curated data found for this task!")
        sys.exit(1)
    
    print(f"      Found {len(data)} curated images")
    
    # Count by label
    label_counts = defaultdict(int)
    for item in data:
        label_counts[item['label']] += 1
    
    print("\n      Class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"        - {label}: {count}")
    
    # Perform stratified split
    print(f"\n[2/4] Performing stratified split ({int(train_ratio*100)}/{int(valid_ratio*100)}/{int(test_ratio*100)})...")
    splits = stratified_split(data, train_ratio, valid_ratio, test_ratio, seed)
    
    for split_name, split_data in splits.items():
        print(f"      {split_name}: {len(split_data)} images")
    
    # Create export directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    export_name = f"{task}_export_{timestamp}"
    export_path = os.path.join(output_dir, export_name)
    
    print(f"\n[3/4] Downloading images to {export_path}...")
    
    # Download images for each split
    download_stats = {'success': 0, 'failed': 0}
    split_counts = {split: defaultdict(int) for split in ['train', 'valid', 'test']}
    
    for split_name, split_data in splits.items():
        split_path = os.path.join(export_path, split_name)
        
        print(f"\n      Downloading {split_name} set ({len(split_data)} images)...")
        
        for item in tqdm(split_data, desc=f"      {split_name}"):
            label = item['label']
            label_path = os.path.join(split_path, label)
            
            # Generate unique filename
            url_hash = hashlib.md5(item['minio_url'].encode()).hexdigest()[:8]
            filename = f"{item['uuid']}_{url_hash}.jpg"
            save_path = os.path.join(label_path, filename)
            
            if download_image(item['minio_url'], save_path):
                download_stats['success'] += 1
                split_counts[split_name][label] += 1
            else:
                download_stats['failed'] += 1
    
    # Generate metadata
    print(f"\n[4/4] Generating metadata...")
    
    metadata = {
        "export_id": export_name,
        "task": task,
        "created_at": datetime.now().isoformat(),
        "source_table": "vlm_curated",
        "excluded_labels": ["unknown"],
        "classes": config['classes'],
        "split_ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio,
        },
        "random_seed": seed,
        "total_images": download_stats['success'],
        "failed_downloads": download_stats['failed'],
        "class_distribution": {
            split_name: dict(counts) 
            for split_name, counts in split_counts.items()
        },
        "paths": {
            "train": os.path.join(export_path, "train"),
            "valid": os.path.join(export_path, "valid"),
            "test": os.path.join(export_path, "test"),
        }
    }
    
    # Calculate statistics
    all_counts = defaultdict(int)
    for split_counts_dict in split_counts.values():
        for label, count in split_counts_dict.items():
            all_counts[label] += count
    
    metadata["total_per_class"] = dict(all_counts)
    
    if all_counts:
        metadata["min_class_count"] = min(all_counts.values())
        metadata["max_class_count"] = max(all_counts.values())
        metadata["imbalance_ratio"] = round(max(all_counts.values()) / max(min(all_counts.values()), 1), 2)
    
    # Save metadata
    metadata_path = os.path.join(export_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  Export Complete!")
    print(f"{'='*60}")
    print(f"  Location: {export_path}")
    print(f"  Total images: {download_stats['success']}")
    print(f"  Failed downloads: {download_stats['failed']}")
    print(f"\n  Split distribution:")
    for split_name, counts in split_counts.items():
        total = sum(counts.values())
        print(f"    {split_name}: {total} images")
        for label, count in sorted(counts.items()):
            print(f"      - {label}: {count}")
    print(f"\n  Metadata: {metadata_path}")
    print(f"{'='*60}\n")
    
    return export_path, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export curated data for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export gender data with default 70/15/15 split
  python export_curated_data.py --task gender --output /data/exports

  # Export age data with custom split
  python export_curated_data.py --task age --output /data/exports --split 80 10 10

  # Export staff data
  python export_curated_data.py --task staff --output /data/exports

Tasks available:
  - gender: male, female (excludes unknown)
  - age: infant, child, teen, adult, mature (excludes unknown)
  - staff: staff, non_staff
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["gender", "age", "staff"],
        help="Task to export data for"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/nihit/highlander/data/exports",
        help="Output directory for exported data"
    )
    
    parser.add_argument(
        "--split",
        type=int,
        nargs=3,
        default=[70, 15, 15],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Split percentages (must sum to 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert split percentages to ratios
    split_ratios = tuple(x / 100.0 for x in args.split)
    
    if sum(args.split) != 100:
        print(f"ERROR: Split percentages must sum to 100, got {sum(args.split)}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run export
    export_data(
        task=args.task,
        output_dir=args.output,
        split_ratios=split_ratios,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

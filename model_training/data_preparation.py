#!/usr/bin/env python3
"""
OLY VISION - Data Preparation for Model Training
-------------------------------------------------
Enhanced data preparation with stratified splitting, augmentation,
and proper handling of train/valid/test sets.

Features:
- Stratified train/valid/test split
- Class balancing through augmentation (train only)
- Excludes unknown labels
- Preserves original images in valid/test sets
- Generates comprehensive statistics

Usage:
    python data_preparation.py --input /data/exports/gender_export --output /data/prepared
    python data_preparation.py --input /data/exports/age_export --output /data/prepared --augment
"""

import os
import sys
import json
import math
import shutil
import random
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("WARNING: albumentations not installed. Augmentation disabled.")
    print("Install with: pip install albumentations")
    A = None


# Augmentation pipeline
def get_augmentation_pipeline():
    """Create augmentation pipeline for training data."""
    if A is None:
        return None
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    ])


def count_images_per_class(root_dir: str) -> dict:
    """Count images in each class folder."""
    counts = {}
    if not os.path.exists(root_dir):
        return counts
    
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            counts[class_name] = len(image_files)
    
    return counts


def augment_class(source_dir: str, target_count: int, save_dir: str = None):
    """
    Augment images in a class folder to reach target count.
    
    Args:
        source_dir: Directory containing original images
        target_count: Target number of images after augmentation
        save_dir: Directory to save augmented images (default: same as source)
    """
    if A is None:
        print("WARNING: albumentations not installed. Skipping augmentation.")
        return 0
    
    if save_dir is None:
        save_dir = source_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get existing images
    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    current_count = len(image_files)
    augment_needed = target_count - current_count
    
    if augment_needed <= 0:
        return 0
    
    # Calculate augmentations per image
    aug_per_image = math.ceil(augment_needed / current_count)
    
    aug_pipeline = get_augmentation_pipeline()
    augmented_count = 0
    
    for img_file in tqdm(image_files, desc=f"Augmenting", leave=False):
        if augmented_count >= augment_needed:
            break
        
        img_path = os.path.join(source_dir, img_file)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
        
        for i in range(1, aug_per_image + 1):
            if augmented_count >= augment_needed:
                break
            
            # Apply augmentation
            augmented = aug_pipeline(image=image_np)
            aug_image = Image.fromarray(augmented['image'])
            
            # Generate filename
            base_name = os.path.splitext(img_file)[0]
            aug_filename = f"{base_name}_aug{i}.jpg"
            aug_path = os.path.join(save_dir, aug_filename)
            
            aug_image.save(aug_path, quality=95)
            augmented_count += 1
    
    return augmented_count


def balance_classes(train_dir: str, strategy: str = "oversample"):
    """
    Balance classes in training set through augmentation.
    
    Args:
        train_dir: Path to training directory
        strategy: 'oversample' to match max class, 'undersample' to match min
    """
    print("\n  Balancing classes in training set...")
    
    counts = count_images_per_class(train_dir)
    
    if not counts:
        print("  No classes found to balance!")
        return
    
    print(f"  Before balancing:")
    for class_name, count in sorted(counts.items()):
        print(f"    - {class_name}: {count}")
    
    if strategy == "oversample":
        target_count = max(counts.values())
        print(f"\n  Target count (max): {target_count}")
        
        for class_name, count in counts.items():
            if count < target_count:
                class_path = os.path.join(train_dir, class_name)
                augmented = augment_class(class_path, target_count)
                print(f"    Augmented {class_name}: +{augmented} images")
    
    # Verify final counts
    final_counts = count_images_per_class(train_dir)
    print(f"\n  After balancing:")
    for class_name, count in sorted(final_counts.items()):
        print(f"    - {class_name}: {count}")


def prepare_data(
    input_dir: str,
    output_dir: str,
    augment: bool = True,
    balance: bool = True,
    copy_mode: bool = True
):
    """
    Prepare data for training.
    
    If input already has train/valid/test structure, it will be copied/moved.
    Otherwise, this assumes data needs to be organized.
    
    Args:
        input_dir: Input directory (from export_curated_data.py)
        output_dir: Output directory for prepared data
        augment: Whether to augment training data
        balance: Whether to balance classes
        copy_mode: If True, copy files; if False, move files
    """
    print(f"\n{'='*60}")
    print(f"  OLY VISION - Data Preparation")
    print(f"{'='*60}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Augment: {augment}")
    print(f"  Balance: {balance}")
    print(f"{'='*60}\n")
    
    # Check if input has expected structure
    train_dir = os.path.join(input_dir, "train")
    valid_dir = os.path.join(input_dir, "valid")
    test_dir = os.path.join(input_dir, "test")
    
    has_splits = os.path.isdir(train_dir) and os.path.isdir(valid_dir)
    
    if not has_splits:
        print("ERROR: Input directory must have train/valid/test structure!")
        print("       Use export_curated_data.py to export data first.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    out_train = os.path.join(output_dir, "train")
    out_valid = os.path.join(output_dir, "valid")
    out_test = os.path.join(output_dir, "test")
    
    # Copy/move data
    print("[1/3] Copying data to output directory...")
    
    operation = shutil.copytree if copy_mode else shutil.move
    
    if os.path.exists(out_train):
        shutil.rmtree(out_train)
    if os.path.exists(out_valid):
        shutil.rmtree(out_valid)
    if os.path.exists(out_test) and os.path.isdir(test_dir):
        shutil.rmtree(out_test)
    
    shutil.copytree(train_dir, out_train)
    print(f"      Copied train: {train_dir} -> {out_train}")
    
    shutil.copytree(valid_dir, out_valid)
    print(f"      Copied valid: {valid_dir} -> {out_valid}")
    
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, out_test)
        print(f"      Copied test: {test_dir} -> {out_test}")
    
    # Copy metadata if exists
    metadata_src = os.path.join(input_dir, "metadata.json")
    if os.path.exists(metadata_src):
        shutil.copy2(metadata_src, os.path.join(output_dir, "metadata_original.json"))
    
    # Balance and augment training data
    if balance and augment:
        print("\n[2/3] Balancing and augmenting training data...")
        balance_classes(out_train, strategy="oversample")
    elif augment:
        print("\n[2/3] Augmenting training data (no balancing)...")
        counts = count_images_per_class(out_train)
        target = int(max(counts.values()) * 1.5)  # 50% more augmentation
        for class_name in counts.keys():
            class_path = os.path.join(out_train, class_name)
            augment_class(class_path, target)
    else:
        print("\n[2/3] Skipping augmentation...")
    
    # Generate final statistics
    print("\n[3/3] Generating statistics...")
    
    stats = {
        "prepared_at": datetime.now().isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "augmented": augment,
        "balanced": balance,
        "splits": {}
    }
    
    for split_name, split_dir in [("train", out_train), ("valid", out_valid), ("test", out_test)]:
        if os.path.isdir(split_dir):
            counts = count_images_per_class(split_dir)
            stats["splits"][split_name] = {
                "path": split_dir,
                "total": sum(counts.values()),
                "per_class": counts
            }
    
    # Get class names
    stats["classes"] = sorted(stats["splits"]["train"]["per_class"].keys())
    stats["num_classes"] = len(stats["classes"])
    
    # Save statistics
    stats_path = os.path.join(output_dir, "preparation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  Preparation Complete!")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {stats['classes']}")
    print(f"\n  Final distribution:")
    for split_name, split_info in stats["splits"].items():
        print(f"    {split_name}: {split_info['total']} images")
        for class_name, count in sorted(split_info["per_class"].items()):
            print(f"      - {class_name}: {count}")
    print(f"\n  Statistics: {stats_path}")
    print(f"{'='*60}\n")
    
    return output_dir, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data with augmentation and balancing
  python data_preparation.py --input /data/exports/gender_export_2026-02-21 --output /data/prepared/gender

  # Prepare without augmentation (for quick testing)
  python data_preparation.py --input /data/exports/age_export --output /data/prepared/age --no-augment

  # Prepare without balancing
  python data_preparation.py --input /data/exports/staff_export --output /data/prepared/staff --no-balance
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (from export_curated_data.py)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for prepared data"
    )
    
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation"
    )
    
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing"
    )
    
    args = parser.parse_args()
    
    prepare_data(
        input_dir=args.input,
        output_dir=args.output,
        augment=not args.no_augment,
        balance=not args.no_balance
    )


if __name__ == "__main__":
    main()

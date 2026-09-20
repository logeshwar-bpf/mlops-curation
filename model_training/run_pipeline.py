#!/usr/bin/env python3
"""
OLY VISION - Complete Training Pipeline
----------------------------------------
End-to-end pipeline for model training:
1. Export curated data from database
2. Prepare data (split, augment, balance)
3. Train model
4. Generate artifacts
5. (Optional) Deploy to production

Usage:
    # Run complete pipeline for gender classification
    python run_pipeline.py --task gender --output-base /home/nihit/highlander

    # Run with custom config
    python run_pipeline.py --task age --config config/age_config.yaml

    # Skip export (use existing data)
    python run_pipeline.py --task staff --skip-export --data-dir /data/prepared/staff
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd: list, description: str) -> bool:
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        return False


def run_pipeline(
    task: str,
    output_base: str,
    config_path: Optional[str] = None,
    skip_export: bool = False,
    skip_prepare: bool = False,
    data_dir: Optional[str] = None,
    split_ratios: tuple = (70, 15, 15),
    no_augment: bool = False,
    deploy: bool = False,
    model: Optional[str] = None,
):
    """
    Run the complete training pipeline.
    
    Args:
        task: Task type (gender, age, staff)
        output_base: Base output directory
        config_path: Path to custom config file
        skip_export: Skip data export step
        skip_prepare: Skip data preparation step
        data_dir: Existing prepared data directory
        split_ratios: Train/valid/test split percentages
        no_augment: Disable augmentation
        deploy: Deploy model after training
        model: Model architecture name (timm model name, e.g., 'convnext_atto.d2_in1k')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'#'*60}")
    print(f"#  OLY VISION - Training Pipeline")
    print(f"#  Task: {task.upper()}")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # Setup directories
    data_exports_dir = os.path.join(output_base, "data", "exports")
    data_prepared_dir = os.path.join(output_base, "data", "prepared")
    artifacts_dir = os.path.join(output_base, "artifacts")
    
    os.makedirs(data_exports_dir, exist_ok=True)
    os.makedirs(data_prepared_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    export_dir = None
    prepared_dir = data_dir
    
    # Step 1: Export curated data
    if not skip_export and not data_dir:
        print("\n" + "="*60)
        print("  STEP 1: Exporting Curated Data")
        print("="*60)
        
        from export_curated_data import export_data
        
        export_dir, metadata = export_data(
            task=task,
            output_dir=data_exports_dir,
            split_ratios=tuple(x/100 for x in split_ratios),
        )
        
        print(f"  Export completed: {export_dir}")
    else:
        print("\n  STEP 1: Skipping export (using existing data)")
    
    # Step 2: Prepare data (augmentation, balancing)
    if not skip_prepare:
        print("\n" + "="*60)
        print("  STEP 2: Preparing Data")
        print("="*60)
        
        from data_preparation import prepare_data
        
        input_dir = export_dir or data_dir
        if not input_dir:
            print("ERROR: No input data directory specified!")
            sys.exit(1)
        
        prepared_dir = os.path.join(data_prepared_dir, f"{task}_{timestamp}")
        
        prepared_dir, stats = prepare_data(
            input_dir=input_dir,
            output_dir=prepared_dir,
            augment=not no_augment,
            balance=not no_augment,
        )
        
        print(f"  Preparation completed: {prepared_dir}")
    else:
        print("\n  STEP 2: Skipping preparation (using existing data)")
        if not prepared_dir:
            prepared_dir = data_dir
    
    # Step 3: Train model
    print("\n" + "="*60)
    print("  STEP 3: Training Model")
    print("="*60)
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        from config_loader import load_config
        config = load_config(config_path)
    else:
        # Use default config for task
        default_config_path = os.path.join(
            os.path.dirname(__file__), "config", f"{task}_config.yaml"
        )
        if os.path.exists(default_config_path):
            from config_loader import load_config
            config = load_config(default_config_path)
        else:
            from config_loader import load_config
            config = load_config(os.path.join(
                os.path.dirname(__file__), "config", "default_config.yaml"
            ))
    
    # Update config with actual paths
    config.data.train_dir = os.path.join(prepared_dir, "train")
    config.data.valid_dir = os.path.join(prepared_dir, "valid")
    config.data.test_dir = os.path.join(prepared_dir, "test")
    config.output.save_dir = artifacts_dir
    config.task = task
    
    # Override model architecture if specified
    if model:
        print(f"  Using custom model: {model}")
        config.model.architecture = model
    
    # Run training
    from trainer import train
    results = train(config)
    
    print(f"\n  Training completed!")
    print(f"  Experiment ID: {results['experiment_id']}")
    print(f"  Best Accuracy: {results['best_val_accuracy']:.4f}")
    if results['test_metrics']:
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    
    # Step 4: Deploy (optional)
    if deploy:
        print("\n" + "="*60)
        print("  STEP 4: Deploying Model")
        print("="*60)
        
        from experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker()
        
        # Create model version
        version = f"v{datetime.now().strftime('%Y%m%d%H%M')}"
        
        tracker.create_model_version(
            task=task,
            version=version,
            experiment_id=results['experiment_id'],
            onnx_path=results['onnx_path'],
            pt_path=os.path.join(results['output_dir'], "checkpoints", "model_best.pt"),
            config_path=os.path.join(results['output_dir'], "config.yaml"),
            class_names=results['class_names'],
            accuracy=results['test_metrics'].get('accuracy') if results['test_metrics'] else results['best_val_accuracy'],
        )
        
        # Activate version
        tracker.activate_model_version(task, version)
        
        tracker.close()
        
        print(f"  Model deployed as version: {version}")
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"#  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"  Task: {task}")
    print(f"  Experiment: {results['experiment_id']}")
    print(f"  Output: {results['output_dir']}")
    if results['onnx_path']:
        print(f"  ONNX Model: {results['onnx_path']}")
    print(f"  Training Time: {results['training_time_seconds']}s")
    print(f"{'#'*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete model training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train gender classifier
  python run_pipeline.py --task gender

  # Train age classifier with custom config
  python run_pipeline.py --task age --config config/age_config.yaml

  # Train staff classifier and deploy
  python run_pipeline.py --task staff --deploy

  # Skip export, use existing data
  python run_pipeline.py --task gender --skip-export --data-dir /data/prepared/gender

  # Run without augmentation (quick test)
  python run_pipeline.py --task gender --no-augment
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["gender", "age", "staff"],
        help="Classification task"
    )
    
    parser.add_argument(
        "--output-base",
        type=str,
        default="/home/nihit/highlander",
        help="Base output directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML file"
    )
    
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip data export step"
    )
    
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip data preparation step"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Existing prepared data directory (use with --skip-export)"
    )
    
    parser.add_argument(
        "--split",
        type=int,
        nargs=3,
        default=[70, 15, 15],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Split percentages (default: 70 15 15)"
    )
    
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy model after training"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture (timm model name, e.g., 'convnext_atto.d2_in1k', 'resnet18', 'efficientnet_b0')"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        task=args.task,
        output_base=args.output_base,
        config_path=args.config,
        skip_export=args.skip_export,
        skip_prepare=args.skip_prepare,
        data_dir=args.data_dir,
        split_ratios=tuple(args.split),
        no_augment=args.no_augment,
        deploy=args.deploy,
        model=args.model,
    )


if __name__ == "__main__":
    main()

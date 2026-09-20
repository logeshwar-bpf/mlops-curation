#!/usr/bin/env python3
"""
OLY VISION - Model Trainer
---------------------------
Enhanced training module with config support, experiment tracking,
and comprehensive artifact generation.

Features:
- YAML config file support
- Experiment tracking in PostgreSQL
- Early stopping
- Learning rate scheduling
- Class weights for imbalanced data
- Automatic artifact generation (confusion matrix, ONNX export, etc.)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: matplotlib not installed. Plots will be disabled.")
    plt = None

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report, 
        accuracy_score, precision_score, recall_score, f1_score
    )
except ImportError:
    print("WARNING: sklearn not installed. Run: pip install scikit-learn")
    sklearn = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("WARNING: albumentations not installed. Using basic transforms.")

from config_loader import Config, load_config
from experiment_tracker import ExperimentTracker, generate_experiment_id, init_tracking_tables


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AlbumentationsTransform:
    """Wrapper for albumentations transforms to work with torchvision datasets."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']


def get_transforms(config: Config, is_train: bool = True):
    """Get data transforms based on config."""
    image_size = config.data.image_size
    aug_config = config.data.augmentation
    
    if is_train and aug_config.get('enabled', True) and HAS_ALBUMENTATIONS:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness_limit', 0.2),
                contrast_limit=aug_config.get('contrast_limit', 0.2),
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=aug_config.get('rotate_limit', 10),
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return AlbumentationsTransform(transform)
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test dataloaders."""
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    train_dataset = datasets.ImageFolder(config.data.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.data.valid_dir, transform=val_transform)
    
    test_dataset = None
    test_loader = None
    if os.path.isdir(config.data.test_dir):
        test_dataset = datasets.ImageFolder(config.data.test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names


def create_model(config: Config, num_classes: int) -> nn.Module:
    """Create model based on config."""
    model = timm.create_model(
        config.model.architecture,
        pretrained=config.model.pretrained,
        num_classes=num_classes
    )
    
    # Replace classifier if custom classifier config is provided
    classifier_config = config.model.classifier
    if classifier_config.get('hidden_dims'):
        num_in_features = model.get_classifier().in_features
        hidden_dims = classifier_config['hidden_dims']
        use_batchnorm = classifier_config.get('use_batchnorm', True)
        dropout = config.model.dropout
        
        layers = []
        prev_dim = num_in_features
        
        for hidden_dim in hidden_dims:
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=not use_batchnorm))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        custom_classifier = nn.Sequential(*layers)
        
        # Replace classifier based on model architecture
        # ConvNeXt and similar models need special handling
        arch_lower = config.model.architecture.lower()
        
        if 'convnext' in arch_lower:
            # ConvNeXt has head.fc as the final classifier, keep the pooling/norm layers
            if hasattr(model, 'head') and hasattr(model.head, 'fc'):
                model.head.fc = custom_classifier
            else:
                # Fallback: reset head with proper pooling
                model.reset_classifier(num_classes=0)  # Remove classifier, keep pooling
                model.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    *layers
                )
        elif 'vit' in arch_lower or 'swin' in arch_lower or 'deit' in arch_lower:
            # Vision Transformers typically use model.head
            model.head = custom_classifier
        elif hasattr(model, 'fc'):
            # ResNet, etc.
            model.fc = custom_classifier
        elif hasattr(model, 'classifier'):
            # EfficientNet, MobileNet, etc.
            model.classifier = custom_classifier
        elif hasattr(model, 'head'):
            # Generic head replacement
            model.head = custom_classifier
        else:
            # Last resort: use timm's reset_classifier
            logging.warning(f"Unknown model architecture {config.model.architecture}, using default classifier")
            model.reset_classifier(num_classes=num_classes)
    
    return model


def get_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Calculate balanced class weights."""
    class_counts = torch.zeros(num_classes)
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return weights.to(device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = None,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
):
    """Save confusion matrix plot."""
    if plt is None:
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """Save learning curves plot."""
    if plt is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int],
    device: torch.device
):
    """Export model to ONNX format."""
    model.eval()
    
    # Add softmax for inference
    model_export = nn.Sequential(model, nn.Softmax(dim=1))
    
    dummy_input = torch.ones(1, 3, input_size[0], input_size[1]).to(device)
    
    torch.onnx.export(
        model_export,
        dummy_input,
        save_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {save_path}")


def train(config: Config) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        config: Training configuration
    
    Returns:
        Dictionary with training results
    """
    # Initialize tracking
    init_tracking_tables()
    tracker = ExperimentTracker()
    
    # Generate experiment ID
    experiment_id = generate_experiment_id(config.task, config.model.architecture)
    
    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output.save_dir, f"{experiment_id}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(output_dir, "config.yaml")
    from config_loader import save_config
    save_config(config, config_save_path)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
    num_classes = len(class_names)
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create experiment record
    tracker.create_experiment(
        experiment_id=experiment_id,
        name=config.experiment_name,
        task=config.task,
        config=config.to_dict(),
        description=config.description,
        class_names=class_names,
        train_images=len(train_loader.dataset),
        valid_images=len(val_loader.dataset),
        test_images=len(test_loader.dataset) if test_loader else 0,
    )
    
    # Create model
    logger.info(f"Creating model: {config.model.architecture}")
    model = create_model(config, num_classes)
    model = model.to(device)
    
    # Loss function with optional class weights
    if config.training.class_weights == "balanced":
        weights = get_class_weights(train_loader, num_classes, device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"Using balanced class weights: {weights.cpu().numpy()}")
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    if config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    
    # Scheduler
    scheduler_config = config.training.scheduler
    scheduler = None
    if scheduler_config.get('type') == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    # Early stopping
    early_stopping = None
    es_config = config.training.early_stopping
    if es_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=es_config.get('patience', 7),
            min_delta=es_config.get('min_delta', 0.001),
            mode='min' if es_config.get('monitor', 'val_loss') == 'val_loss' else 'max'
        )
    
    # Training loop
    logger.info("Starting training...")
    tracker.start_experiment(experiment_id)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    start_time = time.time()
    
    try:
        for epoch in range(1, config.training.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                gradient_clip=config.training.gradient_clip
            )
            
            # Validate
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch}/{config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Log to database
            tracker.log_epoch(
                experiment_id=experiment_id,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                
                if config.output.save_best_only:
                    torch.save(
                        best_model_state,
                        os.path.join(output_dir, "checkpoints", "model_best.pt")
                    )
            
            # Save checkpoint every epoch if configured
            if config.output.save_every_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "checkpoints", f"model_epoch_{epoch}.pt")
                )
            
            # Scheduler step
            if scheduler:
                scheduler.step()
            
            # Early stopping
            if early_stopping:
                monitor_value = val_loss if es_config.get('monitor', 'val_loss') == 'val_loss' else val_acc
                if early_stopping(monitor_value):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        training_time = int(time.time() - start_time)
        logger.info(f"Training completed in {training_time}s")
        
        # Save final model (last epoch) before loading best
        final_model_state = model.state_dict().copy()
        torch.save(final_model_state, os.path.join(output_dir, "checkpoints", "model_final.pt"))
        logger.info("Saved final model checkpoint")
        
        # Load best model for evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_metrics = {}
        if test_loader:
            logger.info("Evaluating on test set...")
            _, test_acc, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device)
            
            test_metrics = {
                'accuracy': accuracy_score(test_labels, test_preds),
                'precision': precision_score(test_labels, test_preds, average='macro'),
                'recall': recall_score(test_labels, test_preds, average='macro'),
                'f1': f1_score(test_labels, test_preds, average='macro'),
            }
            
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            
            # Save confusion matrix
            if config.output.artifacts.get('confusion_matrix', True):
                save_confusion_matrix(
                    test_labels, test_preds, class_names,
                    os.path.join(output_dir, "plots", "confusion_matrix_test.png"),
                    title=f"Test Confusion Matrix (Acc: {test_metrics['accuracy']:.2%})"
                )
            
            # Save classification report
            if config.output.artifacts.get('classification_report', True):
                report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
                with open(os.path.join(output_dir, "classification_report.json"), 'w') as f:
                    json.dump(report, f, indent=2)
        
        # Save learning curves
        if config.output.artifacts.get('learning_curves', True):
            save_learning_curves(
                train_losses, val_losses, train_accs, val_accs,
                os.path.join(output_dir, "plots", "learning_curves.png")
            )
        
        # Export to ONNX (both best and final)
        onnx_path = None
        onnx_final_path = None
        if config.output.artifacts.get('onnx_export', True):
            # Export best model (currently loaded)
            onnx_path = os.path.join(output_dir, f"{config.experiment_name}_best.onnx")
            export_to_onnx(model, onnx_path, config.data.image_size, device)
            logger.info(f"Exported best model to ONNX: {onnx_path}")
            
            # Export final model (last epoch)
            model.load_state_dict(final_model_state)
            onnx_final_path = os.path.join(output_dir, f"{config.experiment_name}_final.onnx")
            export_to_onnx(model, onnx_final_path, config.data.image_size, device)
            logger.info(f"Exported final model to ONNX: {onnx_final_path}")
            
            # Reload best model for any subsequent operations
            if best_model_state:
                model.load_state_dict(best_model_state)
        
        # Save class names mapping
        class_mapping = {
            'classes': class_names,
            'index_to_class': {i: name for i, name in enumerate(class_names)},
            'class_to_index': {name: i for i, name in enumerate(class_names)},
        }
        with open(os.path.join(output_dir, "class_names.json"), 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Update experiment record
        tracker.complete_experiment(
            experiment_id=experiment_id,
            best_epoch=best_epoch,
            best_val_accuracy=val_accs[best_epoch - 1],
            best_val_loss=best_val_loss,
            test_accuracy=test_metrics.get('accuracy'),
            test_precision=test_metrics.get('precision'),
            test_recall=test_metrics.get('recall'),
            test_f1=test_metrics.get('f1'),
            training_time_seconds=training_time,
            artifact_dir=output_dir,
            best_model_path=os.path.join(output_dir, "checkpoints", "model_best.pt"),
            onnx_model_path=onnx_path,
        )
        
        results = {
            'experiment_id': experiment_id,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': val_accs[best_epoch - 1],
            'test_metrics': test_metrics,
            'training_time_seconds': training_time,
            'output_dir': output_dir,
            'onnx_path': onnx_path,
            'onnx_final_path': onnx_final_path,
            'class_names': class_names,
        }
        
        logger.info(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.fail_experiment(experiment_id, str(e))
        raise
    finally:
        tracker.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    results = train(config)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Experiment ID: {results['experiment_id']}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    if results['test_metrics']:
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Output: {results['output_dir']}")
    print("="*60)

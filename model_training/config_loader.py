#!/usr/bin/env python3
"""
OLY VISION - Configuration Loader
----------------------------------
Loads and validates training configuration from YAML files.
"""

import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


@dataclass
class DataConfig:
    train_dir: str
    valid_dir: str
    test_dir: str
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.4
    classifier: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: Dict[str, Any] = field(default_factory=dict)
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    class_weights: str = "balanced"
    gradient_clip: float = 1.0


@dataclass
class OutputConfig:
    save_dir: str = "/artifacts"
    save_every_epoch: bool = False
    save_best_only: bool = True
    artifacts: Dict[str, bool] = field(default_factory=dict)


@dataclass 
class Config:
    """Complete training configuration."""
    experiment_name: str
    task: str
    description: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
    device: str = "auto"
    mixed_precision: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        exp = config_dict.get('experiment', {})
        data = config_dict.get('data', {})
        model = config_dict.get('model', {})
        training = config_dict.get('training', {})
        output = config_dict.get('output', {})
        hardware = config_dict.get('hardware', {})
        logging = config_dict.get('logging', {})
        
        # Convert image_size to tuple
        image_size = data.get('image_size', [224, 224])
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        
        return cls(
            experiment_name=exp.get('name', 'experiment'),
            task=exp.get('task', 'gender'),
            description=exp.get('description', ''),
            data=DataConfig(
                train_dir=data.get('train_dir', ''),
                valid_dir=data.get('valid_dir', ''),
                test_dir=data.get('test_dir', ''),
                image_size=image_size,
                batch_size=data.get('batch_size', 32),
                num_workers=data.get('num_workers', 4),
                pin_memory=data.get('pin_memory', True),
                augmentation=data.get('augmentation', {}),
            ),
            model=ModelConfig(
                architecture=model.get('architecture', 'efficientnet_b0'),
                pretrained=model.get('pretrained', True),
                dropout=model.get('dropout', 0.4),
                classifier=model.get('classifier', {}),
            ),
            training=TrainingConfig(
                epochs=training.get('epochs', 30),
                learning_rate=training.get('learning_rate', 0.0001),
                weight_decay=training.get('weight_decay', 0.0001),
                optimizer=training.get('optimizer', 'adam'),
                scheduler=training.get('scheduler', {}),
                early_stopping=training.get('early_stopping', {}),
                class_weights=training.get('class_weights', 'balanced'),
                gradient_clip=training.get('gradient_clip', 1.0),
            ),
            output=OutputConfig(
                save_dir=output.get('save_dir', '/artifacts'),
                save_every_epoch=output.get('save_every_epoch', False),
                save_best_only=output.get('save_best_only', True),
                artifacts=output.get('artifacts', {}),
            ),
            device=hardware.get('device', 'auto'),
            mixed_precision=hardware.get('mixed_precision', False),
            log_level=logging.get('level', 'INFO'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'experiment': {
                'name': self.experiment_name,
                'task': self.task,
                'description': self.description,
            },
            'data': {
                'train_dir': self.data.train_dir,
                'valid_dir': self.data.valid_dir,
                'test_dir': self.data.test_dir,
                'image_size': list(self.data.image_size),
                'batch_size': self.data.batch_size,
                'num_workers': self.data.num_workers,
                'pin_memory': self.data.pin_memory,
                'augmentation': self.data.augmentation,
            },
            'model': {
                'architecture': self.model.architecture,
                'pretrained': self.model.pretrained,
                'dropout': self.model.dropout,
                'classifier': self.model.classifier,
            },
            'training': {
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'optimizer': self.training.optimizer,
                'scheduler': self.training.scheduler,
                'early_stopping': self.training.early_stopping,
                'class_weights': self.training.class_weights,
                'gradient_clip': self.training.gradient_clip,
            },
            'output': {
                'save_dir': self.output.save_dir,
                'save_every_epoch': self.output.save_every_epoch,
                'save_best_only': self.output.save_best_only,
                'artifacts': self.output.artifacts,
            },
            'hardware': {
                'device': self.device,
                'mixed_precision': self.mixed_precision,
            },
            'logging': {
                'level': self.log_level,
            }
        }


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def validate_config(config: Config) -> bool:
    """Validate configuration values."""
    errors = []
    
    # Check data directories
    if not os.path.isdir(config.data.train_dir):
        errors.append(f"Training directory not found: {config.data.train_dir}")
    if not os.path.isdir(config.data.valid_dir):
        errors.append(f"Validation directory not found: {config.data.valid_dir}")
    
    # Check training parameters
    if config.training.epochs <= 0:
        errors.append(f"Epochs must be positive: {config.training.epochs}")
    if config.training.learning_rate <= 0:
        errors.append(f"Learning rate must be positive: {config.training.learning_rate}")
    if config.training.batch_size <= 0:
        errors.append(f"Batch size must be positive: {config.training.batch_size}")
    
    # Check model architecture
    valid_architectures = [
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'resnet18', 'resnet34', 'resnet50',
        'mobilenetv3_small_100', 'mobilenetv3_large_100',
        'vit_tiny_patch16_224', 'vit_small_patch16_224',
    ]
    if config.model.architecture not in valid_architectures:
        print(f"WARNING: Unknown architecture '{config.model.architecture}'. May still work if available in timm.")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Test loading config
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/default_config.yaml"
    
    try:
        config = load_config(config_path)
        print(f"Loaded config: {config.experiment_name}")
        print(f"Task: {config.task}")
        print(f"Model: {config.model.architecture}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Valid: {validate_config(config)}")
    except Exception as e:
        print(f"Error: {e}")

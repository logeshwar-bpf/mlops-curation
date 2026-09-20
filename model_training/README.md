# OLY VISION - Model Training Pipeline

Complete end-to-end pipeline for training classification models from curated data.

## Features

- **Data Export**: Extract curated images from PostgreSQL (excluding "unknown" labels)
- **Data Preparation**: Stratified splitting, augmentation, class balancing
- **Model Training**: Config-driven training with experiment tracking
- **Artifact Generation**: Confusion matrix, learning curves, ONNX export
- **Experiment Tracking**: PostgreSQL-based tracking of all experiments

## Quick Start

### 1. Install Dependencies

```bash
cd /home/nihit/highlander/model_training
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Train gender classifier
python run_pipeline.py --task gender

# Train age classifier
python run_pipeline.py --task age

# Train staff classifier
python run_pipeline.py --task staff
```

### 3. Deploy Model

```bash
# Train and deploy
python run_pipeline.py --task gender --deploy
```

## Pipeline Steps

### Step 1: Export Curated Data

Extracts labeled images from `vlm_curated` table, **excluding unknown labels**.

```bash
python export_curated_data.py --task gender --output /data/exports
```

**Options:**
- `--task`: gender, age, or staff
- `--split`: Train/valid/test percentages (default: 70 15 15)
- `--seed`: Random seed for reproducibility

**Output:**
```
/data/exports/gender_export_2026-02-21/
├── train/
│   ├── male/
│   └── female/
├── valid/
│   ├── male/
│   └── female/
├── test/
│   ├── male/
│   └── female/
└── metadata.json
```

### Step 2: Prepare Data

Balances classes through augmentation (train set only).

```bash
python data_preparation.py --input /data/exports/gender_export --output /data/prepared/gender
```

**Options:**
- `--no-augment`: Skip augmentation
- `--no-balance`: Skip class balancing

### Step 3: Train Model

```bash
python trainer.py --config config/gender_config.yaml
```

**Output:**
```
/artifacts/gender_efficientnet_b0_20260221_103045/
├── checkpoints/
│   └── model_best.pt
├── plots/
│   ├── confusion_matrix_test.png
│   └── learning_curves.png
├── config.yaml
├── class_names.json
├── classification_report.json
└── gender_classifier_best.onnx
```

## Configuration

Configuration files are in `config/` directory:

- `default_config.yaml` - Base configuration
- `gender_config.yaml` - Gender classification
- `age_config.yaml` - Age classification (5 classes)
- `staff_config.yaml` - Staff detection (binary)

### Key Configuration Options

```yaml
experiment:
  name: "gender_classifier"
  task: "gender"

data:
  train_dir: "/data/prepared/gender/train"
  valid_dir: "/data/prepared/gender/valid"
  test_dir: "/data/prepared/gender/test"
  image_size: [224, 224]
  batch_size: 32

model:
  architecture: "efficientnet_b0"
  pretrained: true
  dropout: 0.4

training:
  epochs: 30
  learning_rate: 0.0001
  class_weights: "balanced"
  early_stopping:
    enabled: true
    patience: 7
```

### Supported Models

From [timm](https://huggingface.co/timm) library:
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- `resnet18`, `resnet34`, `resnet50`
- `mobilenetv3_small_100`, `mobilenetv3_large_100`
- `vit_tiny_patch16_224`, `vit_small_patch16_224`

## Classes by Task

| Task | Classes | Description |
|------|---------|-------------|
| **gender** | male, female | Binary gender (unknown excluded) |
| **age** | infant, child, teen, adult, mature | 5-class age (unknown excluded) |
| **staff** | staff, non_staff | Binary staff detection |

## Experiment Tracking

All experiments are tracked in PostgreSQL:

```sql
-- View recent experiments
SELECT experiment_id, task, status, best_val_accuracy, test_accuracy
FROM training_experiments
ORDER BY created_at DESC
LIMIT 10;

-- View training metrics
SELECT epoch, train_loss, val_loss, val_accuracy
FROM training_metrics
WHERE experiment_id = 'gender_efficientnet_b0_20260221_103045'
ORDER BY epoch;

-- View deployed models
SELECT task, version, accuracy, is_active, deployed_at
FROM model_versions
WHERE is_active = TRUE;
```

## Directory Structure

```
model_training/
├── config/                     # YAML configuration files
│   ├── default_config.yaml
│   ├── gender_config.yaml
│   ├── age_config.yaml
│   └── staff_config.yaml
├── export_curated_data.py      # Export from database
├── data_preparation.py         # Augmentation & balancing
├── config_loader.py            # Config file handling
├── experiment_tracker.py       # Database tracking
├── trainer.py                  # Model training
├── run_pipeline.py             # Complete pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage Examples

### Quick Test (No Augmentation)

```bash
python run_pipeline.py --task gender --no-augment
```

### Custom Split Ratio

```bash
python run_pipeline.py --task age --split 80 10 10
```

### Use Existing Data

```bash
python run_pipeline.py --task staff --skip-export --data-dir /data/prepared/staff
```

### Train Multiple Models

```bash
# Edit config to try different architectures
python trainer.py --config config/gender_config.yaml  # efficientnet_b0
# Change architecture in config
python trainer.py --config config/gender_config.yaml  # resnet18
```

## Deploying Trained Models

After training, the ONNX model can be used in `kafka_consumer.py`:

```python
import onnxruntime as ort
import json

# Load model
session = ort.InferenceSession("gender_classifier_best.onnx")
with open("class_names.json") as f:
    class_map = json.load(f)['index_to_class']

# Inference
def predict(image):
    input_tensor = preprocess(image)  # Resize to 224x224, normalize
    outputs = session.run(None, {'input': input_tensor})
    probs = outputs[0][0]
    class_idx = probs.argmax()
    return class_map[str(class_idx)], probs[class_idx]
```

## Troubleshooting

### "No curated data found"

Ensure you have curated images in `vlm_curated` table:
```sql
SELECT COUNT(*) FROM vlm_curated WHERE gender_category IN ('male', 'female');
```

### Out of Memory

Reduce batch size in config:
```yaml
data:
  batch_size: 16  # Reduce from 32
```

### Slow Training

- Enable GPU: Ensure CUDA is available
- Reduce num_workers if CPU-bound
- Use smaller model (mobilenetv3_small_100)

### Import Errors

Install all dependencies:
```bash
pip install -r requirements.txt
```

# Malicious URL Detection Model

A dual-input hybrid deep learning system for classifying URLs into four categories: **Benign**, **Defacement**, **Phishing**, and **Malware**.

## Project Overview

This project implements a state-of-the-art malicious URL detection system using a dual-branch neural network architecture:

- **Branch A (Sequence Analysis)**: Character-level CNN + Bi-GRU + Attention for deep feature learning
- **Branch B (Heuristic Analysis)**: 23 deterministic lexical features processed through dense layers
- **Classification Head**: Merged representation with 4-class softmax output

**Key Features:**
- 🎯 Targets ≥95% overall accuracy with <5% Phishing→Benign misclassification
- 🔄 Fully reproducible with deterministic preprocessing and seeded training
- 📊 Data-driven max_sequence_length (95th percentile, not hardcoded)
- 💾 Complete artifact persistence for inference without retraining
- 🧪 Comprehensive unit tests for all modules

## Project Structure

```
major project/
├── src/
│   ├── data_loader.py            # Dataset loading and splitting
│   ├── feature_engineering.py    # 23 lexical feature extractors
│   ├── text_processing.py        # Character-level tokenization
│   ├── model_builder.py          # Dual-input Keras model
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation with metrics
│   ├── classify_url.py           # Single-URL inference
│   └── utils.py                  # Shared utilities
├── tests/                        # Unit tests
├── artifacts/                    # Saved models & artifacts (gitignored)
├── datasets/                     # Training data (gitignored)
├── config.yaml                   # All hyperparameters
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### 1. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset `malicious_phish.csv` should be placed in the `datasets/` folder.

**Dataset details:**
- Source: Kaggle malicious URLs dataset
- Size: 651,191 URL records
- Classes: benign (428K), defacement (96K), phishing (94K), malware (33K)
- Format: CSV with columns `url` and `type`

```powershell
# Ensure dataset is in the correct location:
# datasets/malicious_phish.csv
```

The config is already set to: `../datasets/malicious_phish.csv`

### 3. Verify GPU (Optional but Recommended)

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage

### Training the Model

```powershell
cd src
python train.py
```

**Output artifacts:**
- `artifacts/model.keras` - Trained model
- `artifacts/scaler.pkl` - Feature scaler
- `artifacts/tokenizer.json` - Character tokenizer
- `artifacts/metadata.json` - Training metadata
- `artifacts/results/training_curves.png` - Loss/accuracy plots

**Expected training time:** ≤2 hours on GPU (NVIDIA RTX 3060 or better)

### Evaluating the Model

```powershell
cd src
python evaluate.py
```

**Generated reports:**
- `artifacts/results/classification_report.txt` - Per-class metrics
- `artifacts/results/confusion_matrix.png` - Visualization with Phishing→Benign highlight

**Success criteria checks:**
- ✓ Overall accuracy ≥ 95%
- ✓ Phishing F1-score ≥ 0.93
- ✓ Phishing→Benign misclassification ≤ 5%

### Classifying Individual URLs

```powershell
cd src
python classify_url.py --url "http://example.com/suspicious-login"
```

**Example output:**
```
Predicted Class: PHISHING
Confidence:      87.34%

Class Probabilities:
  benign         :   8.21%
  defacement     :   2.15%
  phishing       :  87.34%
  malware        :   2.30%
```

**Quiet mode** (returns only class name):
```powershell
python classify_url.py --url "http://example.com" --quiet
```

### Running Unit Tests

```powershell
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_feature_engineering.py -v
```

## Configuration

All hyperparameters are in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 32 | Character embedding dimension |
| `cnn_filters` | 128 | Conv1D filters in Branch A |
| `gru_units` | 64 | Bi-GRU units in Branch A |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 256 | Training batch size |
| `early_stopping_patience` | 5 | Epochs to wait before early stopping |
| `random_seed` | 42 | Seed for reproducibility |

## Data Pipeline

### Branch A: Character Sequences
1. Fit character-level tokenizer on training URLs
2. Compute `max_sequence_length` from 95th percentile
3. Tokenize and pad sequences (post-padding)

### Branch B: Lexical Features (23 total)
1. Length features: URL, hostname, path
2. Character counts: `.`, `-`, `@`, `?`, `&`, `=`, `_`, `~`, `%`, `*`, `:`
3. Substring counts: `www`, `https`, `http`, digits, letters, directories
4. Boolean flags: IP address, URL shortener detection

All features are deterministic and scaled with `StandardScaler`.

## Model Architecture

```
Input A (Sequences)          Input B (Features)
       ↓                            ↓
   Embedding(32)              Dense(64, relu)
       ↓                            ↓
   Conv1D(128, 3)             Dropout(0.3)
       ↓                            ↓
   MaxPooling(2)               Dense(32, relu)
       ↓
  Bi-GRU(64)
       ↓
   Attention
       ↓
       └──────── Concatenate ───────┘
                     ↓
              Dense(128, relu)
                     ↓
              Dropout(0.5)
                     ↓
            Dense(4, softmax)
```

## Spec-Kit Workflow

This project was developed using [GitHub Spec-Kit](https://github.com/github/spec-kit) methodology:

1. **Constitution** (`.specify/memory/constitution.md`) - Project principles
2. **Specification** (`specs/001-malicious-url-detection/spec.md`) - Requirements
3. **Plan** (`specs/001-malicious-url-detection/plan.md`) - Architecture decisions
4. **Tasks** (`specs/001-malicious-url-detection/tasks.md`) - Implementation roadmap

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | ≥ 95% | On test set (15% of data) |
| Phishing F1-Score | ≥ 0.93 | Critical for phishing detection |
| Phishing→Benign | ≤ 5% | Misclassification rate |
| Training Time | ≤ 2 hours | On NVIDIA RTX 3060 or better |
| Zero NaN Features | 100% | All 23 features must be deterministic |

## Troubleshooting

**GPU not detected:**
```powershell
# Check CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Out of memory during training:**
- Reduce `batch_size` in `config.yaml` (try 128 or 64)
- Close other GPU-intensive applications

**File not found errors:**
- Verify dataset path in `config.yaml`
- Ensure you're running scripts from the `src/` directory

## License

This project is developed for academic research purposes (PHD_EEE6553).

## Contact

For questions or issues, please refer to the project specification documents in `specs/001-malicious-url-detection/`.

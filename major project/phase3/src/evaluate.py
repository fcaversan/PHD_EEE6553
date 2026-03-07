"""
Evaluation pipeline for Malicious URL Detection Model
Classification report, confusion matrix, and performance metrics
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

# Ensure src/ is on the path and working directory is the project root
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

# Import local modules
from utils import set_all_seeds, get_config
from data_loader import load_malicious_urls_dataset
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import tokenize_and_pad, load_tokenizer


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str, 
                         highlight_phishing_benign: bool = True) -> None:
    """
    Generate and save confusion matrix heatmap with optional highlighting.
    
    Args:
        cm: Confusion matrix (N_classes, N_classes)
        class_names: List of class names for labels
        save_path: Path to save the plot
        highlight_phishing_benign: If True, annotate Phishing→Benign cell
    """
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix - Malicious URL Detection', fontsize=15, fontweight='bold', pad=20)
    
    # Highlight Phishing → Benign misclassification if requested
    if highlight_phishing_benign and 'Phishing' in class_names and 'Benign' in class_names:
        try:
            phishing_idx = class_names.index('phishing')
            benign_idx = class_names.index('benign')
            
            # Add red border around Phishing→Benign cell
            from matplotlib.patches import Rectangle
            ax = plt.gca()
            rect = Rectangle(
                (benign_idx, phishing_idx), 1, 1,
                fill=False, edgecolor='red', linewidth=3
            )
            ax.add_patch(rect)
            
            # Add annotation
            phishing_to_benign = cm[phishing_idx, benign_idx]
            total_phishing = cm[phishing_idx, :].sum()
            misclass_rate = (phishing_to_benign / total_phishing * 100) if total_phishing > 0 else 0
            
            plt.text(
                0.5, -0.15,
                f'Phishing→Benign Misclassification: {phishing_to_benign:,} ({misclass_rate:.2f}%)',
                transform=ax.transAxes,
                ha='center',
                fontsize=11,
                color='red',
                fontweight='bold'
            )
        except (ValueError, IndexError):
            # Class names don't match expected format
            pass
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_model():
    """
    Complete evaluation pipeline:
    1. Load trained model and artifacts
    2. Load test data
    3. Generate predictions
    4. Compute and save classification report
    5. Plot and save confusion matrix
    6. Analyze Phishing vs Benign misclassification
    """
    
    print("\n" + "="*60)
    print("MALICIOUS URL DETECTION MODEL - EVALUATION PIPELINE")
    print("="*60)
    
    # Load configuration
    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])
    
    # ========== Load artifacts ==========
    print(f"\n{'='*60}")
    print("Phase 1: Loading Artifacts")
    print(f"{'='*60}")
    
    model_path = config['data']['model_path']
    scaler_path = config['data']['scaler_path']
    tokenizer_path = config['data']['tokenizer_path']
    metadata_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    max_sequence_length = metadata['max_sequence_length']
    label_classes = metadata['label_classes']
    n_lexical_features = int(metadata.get('n_lexical_features', 27))
    print(f"✓ Metadata loaded (max_sequence_length={max_sequence_length})")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # ========== Load test data ==========
    print(f"\n{'='*60}")
    print("Phase 2: Loading Test Data")
    print(f"{'='*60}")
    
    dataset_path = config['data']['dataset_path']
    X_train, y_train, X_val, y_val, X_test, y_test, _ = \
        load_malicious_urls_dataset(dataset_path, config)
    
    # ========== Preprocess test data ==========
    print(f"\n{'='*60}")
    print("Phase 3: Preprocessing Test Data")
    print(f"{'='*60}")
    
    # Extract and scale lexical features
    test_features = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_features, scaler_path)
    if X_test_scaled.shape[1] != n_lexical_features:
        raise ValueError(
            f"Lexical feature mismatch: expected {n_lexical_features}, "
            f"got {X_test_scaled.shape[1]}"
        )
    
    # Tokenize and pad sequences
    X_test_sequences = tokenize_and_pad(tokenizer, X_test, max_sequence_length)
    
    # ========== Generate predictions ==========
    print(f"\n{'='*60}")
    print("Phase 4: Generating Predictions")
    print(f"{'='*60}")
    
    print(f"Running inference on {len(X_test)} test samples...")
    y_pred_probs = model.predict([X_test_sequences, X_test_scaled], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print(f"✓ Predictions generated")
    
    # ========== Compute metrics ==========
    print(f"\n{'='*60}")
    print("Phase 5: Computing Metrics")
    print(f"{'='*60}")
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=label_classes,
        digits=4
    )
    
    print("\n" + report)
    
    # Save classification report
    report_path = os.path.join(config['data']['results_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("MALICIOUS URL DETECTION MODEL - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Classes: {label_classes}\n")
    
    print(f"✓ Classification report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    cm_path = os.path.join(config['data']['results_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_classes, cm_path, highlight_phishing_benign=True)
    
    # ========== Analyze Phishing vs Benign misclassification ==========
    print(f"\n{'='*60}")
    print("Phase 6: Phishing vs Benign Analysis")
    print(f"{'='*60}")
    
    try:
        phishing_idx = label_classes.index('phishing')
        benign_idx = label_classes.index('benign')
        
        phishing_to_benign = cm[phishing_idx, benign_idx]
        total_phishing = cm[phishing_idx, :].sum()
        misclass_rate = (phishing_to_benign / total_phishing * 100) if total_phishing > 0 else 0
        
        print(f"  Total Phishing samples:          {total_phishing}")
        print(f"  Phishing → Benign misclassified: {phishing_to_benign}")
        print(f"  Misclassification rate:          {misclass_rate:.2f}%")
        
        # Success criteria check
        if misclass_rate <= 5.0:
            print(f"  ✓ SUCCESS CRITERION MET: Phishing→Benign rate ≤ 5%")
        else:
            print(f"  ✗ WARNING: Phishing→Benign rate > 5% threshold")
        
    except (ValueError, IndexError):
        print(f"  Note: Could not find 'phishing' or 'benign' in classes: {label_classes}")
    
    # ========== Overall accuracy ==========
    print(f"\n{'='*60}")
    print("Phase 7: Overall Performance")
    print(f"{'='*60}")
    
    test_results = model.evaluate(
        x=[X_test_sequences, X_test_scaled],
        y=y_test,
        verbose=0
    )
    
    print(f"  Test Loss:      {test_results[0]:.4f}")
    print(f"  Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"  Test Precision: {test_results[2]:.4f}")
    print(f"  Test Recall:    {test_results[3]:.4f}")
    
    # Success criteria checks
    print(f"\n{'='*60}")
    print("Success Criteria Validation")
    print(f"{'='*60}")
    
    accuracy = test_results[1]
    if accuracy >= 0.95:
        print(f"  ✓ SC-001: Overall accuracy ≥ 95% ({accuracy*100:.2f}%)")
    else:
        print(f"  ✗ SC-001: Overall accuracy < 95% ({accuracy*100:.2f}%)")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved:")
    print(f"  - Classification report: {report_path}")
    print(f"  - Confusion matrix:      {cm_path}")


if __name__ == '__main__':
    evaluate_model()

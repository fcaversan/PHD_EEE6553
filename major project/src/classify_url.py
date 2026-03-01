"""
Single-URL classifier for Malicious URL Detection Model
Loads saved artifacts and classifies a single URL from command line
"""

import argparse
import json
import numpy as np
from tensorflow import keras

# Import local modules
from utils import get_config
from feature_engineering import extract_lexical_features, load_and_apply_scaler
from text_processing import load_tokenizer, tokenize_and_pad


def classify_url(url: str, verbose: bool = True) -> dict:
    """
    Classify a single URL using trained model and saved artifacts.
    
    Args:
        url: Raw URL string to classify
        verbose: If True, print detailed output
    
    Returns:
        dict: Classification results with predicted class and probabilities
    """
    
    # Load configuration
    config = get_config('config.yaml')
    
    # Load metadata
    metadata_path = config['data']['artifacts_dir'] + '/metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    max_sequence_length = metadata['max_sequence_length']
    label_classes = metadata['label_classes']
    
    if verbose:
        print(f"\n{'='*60}")
        print("Single-URL Classification")
        print(f"{'='*60}")
        print(f"URL: {url}\n")
    
    # Load model
    model_path = config['data']['model_path']
    if verbose:
        print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load tokenizer
    tokenizer_path = config['data']['tokenizer_path']
    tokenizer = load_tokenizer(tokenizer_path)
    
    # ========== Preprocess URL ==========
    
    # Branch A: Tokenize and pad sequence
    url_array = np.array([url])
    sequence = tokenize_and_pad(tokenizer, url_array, max_sequence_length)
    
    # Branch B: Extract lexical features
    features_dict = extract_lexical_features(url)
    features = np.array([list(features_dict.values())], dtype=np.float32)
    
    # Load and apply scaler
    scaler_path = config['data']['scaler_path']
    scaled_features = load_and_apply_scaler(
        features.reshape(1, -1), 
        scaler_path
    )
    
    # ========== Run inference ==========
    predictions = model.predict([sequence, scaled_features], verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = label_classes[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # ========== Format results ==========
    results = {
        'url': url,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            label_classes[i]: float(predictions[0][i])
            for i in range(len(label_classes))
        }
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Classification Results")
        print(f"{'='*60}")
        print(f"Predicted Class: {predicted_class.upper()}")
        print(f"Confidence:      {confidence*100:.2f}%")
        print(f"\nClass Probabilities:")
        for class_name, prob in results['probabilities'].items():
            print(f"  {class_name:15s}: {prob*100:6.2f}%")
        print(f"{'='*60}\n")
    
    return results


def main():
    """
    Command-line interface for single-URL classification.
    """
    
    parser = argparse.ArgumentParser(
        description='Classify a single URL as Benign, Defacement, Phishing, or Malware',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python classify_url.py --url "http://example.com/login"
  python classify_url.py --url "http://192.168.1.1/phishing" --quiet
        '''
    )
    
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL to classify'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output, only show predicted class'
    )
    
    args = parser.parse_args()
    
    # Classify URL
    results = classify_url(args.url, verbose=not args.quiet)
    
    # If quiet mode, just print the predicted class
    if args.quiet:
        print(results['predicted_class'])


if __name__ == '__main__':
    main()

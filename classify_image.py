import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Class names (must match training order)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Model path
MODEL_PATH = 'intel_cnn_model.keras'  # Use existing model file

def load_image_from_url(url_or_path):
    """
    Load image from URL (web or local file path)
    Returns PIL Image object
    """
    # Check if it's a URL (starts with http:// or https://)
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        print(f"📥 Downloading image from: {url_or_path}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url_or_path, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        # Local file path
        if not os.path.exists(url_or_path):
            raise FileNotFoundError(f"Image file not found: {url_or_path}")
        print(f"📂 Loading local image: {url_or_path}")
        img = Image.open(url_or_path)
    
    # Convert to RGB if needed (handles PNG with alpha channel)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess image for model input
    - Resize to target size
    - Convert to numpy array
    - Normalize to [0, 1]
    Returns: preprocessed image array with batch dimension
    """
    # Resize
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(img_resized)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, img_resized

def classify_image(model, img_batch):
    """
    Classify image using the model
    Returns: predicted class index, probabilities
    """
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    probabilities = predictions[0]
    
    return predicted_class, probabilities

def display_results(img, predicted_class, probabilities):
    """
    Display image with classification results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {CLASS_NAMES[predicted_class]}\n'
                  f'Confidence: {probabilities[predicted_class]*100:.2f}%',
                  fontsize=14, fontweight='bold')
    
    # Display probability bar chart
    colors = ['green' if i == predicted_class else 'lightblue' 
              for i in range(len(CLASS_NAMES))]
    bars = ax2.barh(CLASS_NAMES, probabilities * 100, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
def main():
    """
    Main function
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print(f"Please train the model first by running: train_intel_cnn.py")
        sys.exit(1)
    
    # Load model
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"📋 Classes: {', '.join(CLASS_NAMES)}")
    print("=" * 80)
    
    # Get image URL from command line or prompt user
    if len(sys.argv) > 1:
        image_url = sys.argv[1]
    else:
        print("\nEnter image URL (web URL or local file path):")
        print("Example web URL: https://example.com/image.jpg")
        print("Example local: datasets/Image classification-Intel datset/new_Intel_testing_dataset/forest/1.jpg")
        image_url = input("\n🖼️  Image URL/Path: ").strip()
    
    if not image_url:
        print("❌ No image URL provided!")
        sys.exit(1)
    
    try:
        # Load image
        img = load_image_from_url(image_url)
        print(f"✅ Image loaded: {img.size[0]}x{img.size[1]} pixels")
        
        # Preprocess
        print("🔄 Preprocessing image...")
        img_batch, img_resized = preprocess_image(img)
        
        # Classify
        print("🔄 Classifying...")
        predicted_class, probabilities = classify_image(model, img_batch)
        
        # Print results
        print("\n" + "=" * 80)
        print("🎯 CLASSIFICATION RESULTS")
        print("=" * 80)
        print(f"Predicted Class: {CLASS_NAMES[predicted_class].upper()}")
        print(f"Confidence: {probabilities[predicted_class]*100:.2f}%")
        print("\nAll Class Probabilities:")
        for i, (class_name, prob) in enumerate(sorted(
            zip(CLASS_NAMES, probabilities), 
            key=lambda x: x[1], 
            reverse=True
        )):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{emoji} {class_name:>10s}: {prob*100:5.2f}%")
        print("=" * 80)
        
        # Display visual results
        display_results(img_resized, predicted_class, probabilities)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading image: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Visualize Correct and Incorrect Predictions
Shows sample images that are correctly classified vs misclassified
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# SETUP
# ============================================================================
# CUDA Path (required for Windows GPU)
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# Configuration
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'
MODEL_PATH = 'pcb_cnn_model.keras'
IMG_SIZE = (224, 224)  # Must match training (model expects 224x224)
BATCH_SIZE = 32

print("=" * 80)
print("🔍 PREDICTION VISUALIZATION")
print("=" * 80)

# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"\n📦 Loading model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ============================================================================
# LOAD TEST DATA
# ============================================================================
print(f"\n📂 Loading test dataset...")

# Load test data WITHOUT batching for easier indexing
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False  # Keep order for file paths
)

class_names = test_ds.class_names
print(f"✅ Test dataset loaded")
print(f"   Classes: {class_names}")

# ============================================================================
# GET ALL PREDICTIONS
# ============================================================================
print("\n🔮 Generating predictions...")

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds_normalized = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Get all images, labels, and predictions
all_images = []
all_true_labels = []
all_predictions = []

for images, labels in test_ds_normalized:
    preds = model.predict(images, verbose=0)
    all_images.extend(images.numpy())
    all_true_labels.extend(np.argmax(labels.numpy(), axis=1))
    all_predictions.extend(np.argmax(preds, axis=1))

all_images = np.array(all_images)
all_true_labels = np.array(all_true_labels)
all_predictions = np.array(all_predictions)

print(f"✅ Predictions complete")
print(f"   Total test images: {len(all_images)}")

# ============================================================================
# FIND CORRECT AND INCORRECT PREDICTIONS
# ============================================================================
correct_mask = all_true_labels == all_predictions
incorrect_mask = ~correct_mask

correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(incorrect_mask)[0]

n_correct = len(correct_indices)
n_incorrect = len(incorrect_indices)
accuracy = n_correct / len(all_images) * 100

print(f"\n📊 Results:")
print(f"   Correct:   {n_correct}/{len(all_images)} ({accuracy:.1f}%)")
print(f"   Incorrect: {n_incorrect}/{len(all_images)} ({100-accuracy:.1f}%)")

# ============================================================================
# VISUALIZE EXAMPLES
# ============================================================================
print("\n🖼️  Creating visualization...")

# Select examples
n_examples = min(3, n_correct)  # Show up to 3 correct
n_wrong = min(3, n_incorrect)    # Show up to 3 incorrect

fig, axes = plt.subplots(2, max(n_examples, n_wrong), figsize=(15, 8))
if max(n_examples, n_wrong) == 1:
    axes = axes.reshape(2, 1)

# Plot correctly classified images
for i in range(n_examples):
    idx = correct_indices[i]
    ax = axes[0, i]
    
    # Display image
    ax.imshow(all_images[idx])
    true_label = class_names[all_true_labels[idx]]
    pred_label = class_names[all_predictions[idx]]
    
    ax.set_title(f'✅ CORRECT\nTrue: {true_label}\nPred: {pred_label}', 
                 color='green', fontweight='bold', fontsize=10)
    ax.axis('off')

# Fill remaining slots if needed
for i in range(n_examples, max(n_examples, n_wrong)):
    axes[0, i].axis('off')

# Plot misclassified images
for i in range(n_wrong):
    idx = incorrect_indices[i]
    ax = axes[1, i]
    
    # Display image
    ax.imshow(all_images[idx])
    true_label = class_names[all_true_labels[idx]]
    pred_label = class_names[all_predictions[idx]]
    
    ax.set_title(f'❌ WRONG\nTrue: {true_label}\nPred: {pred_label}', 
                 color='red', fontweight='bold', fontsize=10)
    ax.axis('off')

# Fill remaining slots if needed
for i in range(n_wrong, max(n_examples, n_wrong)):
    axes[1, i].axis('off')

plt.suptitle(f'PCB Defect Classification - Examples\nOverall Accuracy: {accuracy:.1f}%', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: prediction_examples.png")
plt.show()

# ============================================================================
# DETAILED BREAKDOWN BY CLASS
# ============================================================================
print("\n" + "=" * 80)
print("📋 PER-CLASS BREAKDOWN")
print("=" * 80)

for i, class_name in enumerate(class_names):
    class_mask = all_true_labels == i
    class_correct = np.sum((all_true_labels == i) & (all_predictions == i))
    class_total = np.sum(class_mask)
    class_acc = class_correct / class_total * 100 if class_total > 0 else 0
    
    # What are they being misclassified as?
    misclassified_as = all_predictions[class_mask & incorrect_mask]
    
    print(f"\n{class_name}:")
    print(f"   Accuracy: {class_correct}/{class_total} ({class_acc:.1f}%)")
    
    if len(misclassified_as) > 0:
        print(f"   Misclassified as:")
        for j in range(len(class_names)):
            if j != i:
                count = np.sum(misclassified_as == j)
                if count > 0:
                    print(f"      - {class_names[j]}: {count} times")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("=" * 80)

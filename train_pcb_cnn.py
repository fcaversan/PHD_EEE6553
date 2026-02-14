"""
PCB Defect Classification - Standalone Training Script
GPU-Optimized version with data caching and prefetching
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import time

# ============================================================================
# GPU SETUP
# ============================================================================
print("=" * 80)
print("🎮 GPU SETUP")
print("=" * 80)

# CUDA Path (required for Windows GPU)
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected!")
    for gpu in gpus:
        print(f"   🚀 {gpu}")
        # Enable memory growth
        tf.config.experimental.set_memory_growth(gpu, True)
    print("   Memory growth enabled")
else:
    print("⚠️  No GPU detected - training will use CPU")

print(f"\nTensorFlow version: {tf.__version__}")
print("=" * 80)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 32  # Balanced batch size
IMG_SIZE = (224, 224)  # Good balance of detail and training stability
EPOCHS = 100  # Quick test
LEARNING_RATE = 0.0005  # Moderate learning rate

# Dataset paths
TRAIN_DIR = 'datasets/PCB_ClassData/Training-PCB'
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'

print(f"\n📊 TRAINING CONFIGURATION")
print(f"   Batch size:     {BATCH_SIZE}")
print(f"   Image size:     {IMG_SIZE}")
print(f"   Epochs:         {EPOCHS}")
print(f"   Learning rate:  {LEARNING_RATE}")
print("=" * 80)

# ============================================================================
# LOAD AND OPTIMIZE DATASET
# ============================================================================
print("\n🔥 Loading dataset with optimized tf.data pipeline...")

# Create tf.data.Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Get class information
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Dataset loaded")
print(f"   Classes: {class_names}")
print(f"   Number of classes: {num_classes}")

# Data normalization (0-255 → 0-1)
normalization_layer = layers.Rescaling(1./255)

# Data augmentation - LIGHTER to preserve feature details
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),  # Reduced - PCB orientation matters
    layers.RandomZoom(0.1),      # Reduced - preserve defect scale
    layers.RandomContrast(0.1),  # Reduced - preserve defect appearance
], name='data_augmentation')

# Apply performance optimizations
print("\n⚡ Applying performance optimizations...")
print("   • Normalizing images (0-255 → 0-1)")
print("   • Data augmentation (flip, rotate, zoom, contrast)")
print("   • Caching dataset in RAM (no disk I/O)")
print("   • Enabling prefetch (GPU never waits)")

# Apply augmentation + normalization to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y))
train_ds = train_ds.cache()  # Load once, reuse every epoch
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Test data: only normalize, no augmentation
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("✅ Optimizations applied")
print("=" * 80)

# ============================================================================
# BUILD MODEL WITH BATCH NORMALIZATION
# ============================================================================
print("\n🏗️  Building CNN model with BatchNorm...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 4
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Light dropout here

    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Reduced from 0.5
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),  # Light dropout
    layers.Dense(num_classes, activation='softmax')
], name='PCB_CNN')

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built and compiled")
print("\n📋 Model Summary:")
model.summary()
print("=" * 80)

# ============================================================================
# COMPUTE CLASS WEIGHTS (Important for balanced training!)
# ============================================================================
print("\n⚖️  Computing class weights to balance training...")

# Count samples per class
class_counts = {}
for i, class_name in enumerate(class_names):
    class_path = os.path.join(TRAIN_DIR, class_name)
    n_samples = len(glob.glob(os.path.join(class_path, '*.*')))
    class_counts[i] = n_samples
    print(f"   {class_name}: {n_samples} samples")

# Compute balanced class weights
total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (num_classes * count) 
                 for i, count in class_counts.items()}

print("\n   Computed class weights:")
for i, class_name in enumerate(class_names):
    print(f"   {class_name}: {class_weights[i]:.3f}")

print("=" * 80)

# ============================================================================
# TRAIN MODEL
# ============================================================================
print(f"\n🚀 Starting GPU-optimized training ({EPOCHS} epochs)...")
print("=" * 80)

start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    class_weight=class_weights,  # Use balanced weights!
    verbose=1
)

elapsed = time.time() - start_time

print("=" * 80)
print(f"✅ TRAINING COMPLETED!")
print(f"   Total time:     {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
print(f"   Time per epoch: {elapsed/EPOCHS:.2f} seconds")
print("=" * 80)

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n📈 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"   Test Loss:     {test_loss:.4f}")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print("=" * 80)

# ============================================================================
# SAVE MODEL (DO THIS EARLY!)
# ============================================================================
model_path = 'pcb_cnn_model.keras'
model.save(model_path)
print(f"\n💾 Model saved: {model_path}")

# ============================================================================
# PREDICTIONS AND CONFUSION MATRIX
# ============================================================================
print("\n🔮 Generating predictions...")

y_pred = []
y_true = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - PCB Defect Classification\nTest Accuracy: {test_acc*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Confusion matrix saved: confusion_matrix.png")
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
print("\n📉 Plotting training history...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training history saved: training_history.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TRAINING SUMMARY")
print("=" * 80)
print(f"   Model:          {model.name}")
print(f"   Parameters:     {model.count_params():,}")
print(f"   Training time:  {elapsed/60:.1f} minutes")
print(f"   Epochs:         {EPOCHS}")
print(f"   Batch size:     {BATCH_SIZE}")
print(f"   Image size:     {IMG_SIZE}")
print(f"   Test accuracy:  {test_acc*100:.2f}%")
print(f"   Model saved:    {model_path}")
print("=" * 80)
print("✅ Done!")

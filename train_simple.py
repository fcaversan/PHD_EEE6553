"""
PCB Defect Classification - Simple CNN
Clean, basic model ready for more data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

# ============================================================================
# GPU SETUP
# ============================================================================
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ============================================================================
# HYPERPARAMETERS - Easy to adjust!
# ============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_DIR = 'datasets/PCB_ClassData/Training-PCB'
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'

print("=" * 80)
print("🎯 SIMPLE CNN TRAINING")
print("=" * 80)
print(f"Image size:    {IMG_SIZE}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Epochs:        {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading dataset...")

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

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Classes: {class_names}")

# Simple preprocessing
normalization = layers.Rescaling(1./255)

# Light data augmentation
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply to training data
train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

# Only normalize test data
test_ds = test_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

print("✅ Data ready")

# ============================================================================
# BUILD SIMPLE CNN
# ============================================================================
print("\n🏗️  Building model...")

model = keras.Sequential([
    # Input
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model ready ({model.count_params():,} parameters)")
model.summary()

# ============================================================================
# TRAIN
# ============================================================================
print("\n🚀 Training...")
print("=" * 80)

start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    verbose=1
)

elapsed = time.time() - start_time
print("=" * 80)
print(f"✅ Training done in {elapsed/60:.1f} minutes")

# ============================================================================
# EVALUATE
# ============================================================================
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n📊 Test Accuracy: {test_acc*100:.2f}%")

# Save model
model.save('simple_model.keras')
print("💾 Model saved: simple_model.keras")

# ============================================================================
# RESULTS
# ============================================================================
print("\n📈 Generating predictions...")

y_pred = []
y_true = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix\nAccuracy: {test_acc*100:.2f}%', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("✅ Saved: confusion_matrix.png")
plt.close()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("✅ Saved: training_history.png")
plt.close()

print("\n" + "=" * 80)
print("🎯 SUMMARY")
print("=" * 80)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Training Time: {elapsed/60:.1f} minutes")
print(f"Model:         simple_model.keras")
print("=" * 80)
print("✅ Done!")

"""
PCB Defect Classification - Regularized CNN
Fighting overfitting with dropout, augmentation, and L2 regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
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
# HYPERPARAMETERS - Tuned to prevent overfitting
# ============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Start with 20 to see progress
LEARNING_RATE = 0.0005  # Lower LR for stability
L2_REG = 0.001  # L2 regularization on conv layers

TRAIN_DIR = 'datasets/PCB_ClassData/Training-PCB'
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'

print("=" * 80)
print("🎯 REGULARIZED CNN TRAINING - Anti-Overfitting Mode")
print("=" * 80)
print(f"Image size:      {IMG_SIZE}")
print(f"Batch size:      {BATCH_SIZE}")
print(f"Epochs:          {EPOCHS}")
print(f"Learning rate:   {LEARNING_RATE}")
print(f"L2 Reg:          {L2_REG}")
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

# Normalization
normalization = layers.Rescaling(1./255)

# AGGRESSIVE data augmentation to prevent memorization
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),  # More rotation
    layers.RandomZoom(0.2),      # More zoom
    layers.RandomContrast(0.2),  # Vary contrast
    layers.RandomBrightness(0.1), # Vary brightness
])

# Apply to training data
train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

# Only normalize test data
test_ds = test_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

print("✅ Data ready with aggressive augmentation")

# ============================================================================
# BUILD REGULARIZED CNN - Simpler + More Dropout + L2
# ============================================================================
print("\n🏗️  Building regularized model...")

model = keras.Sequential([
    # Input
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Block 1 - with L2 regularization
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # Add dropout early
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REG)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Simpler classifier - less capacity = less memorization
    layers.GlobalAveragePooling2D(),  # Instead of Flatten (fewer params)
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(L2_REG)),
    layers.Dropout(0.5),  # Heavy dropout before output
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
# EARLY STOPPING - Stop if validation doesn't improve
# ============================================================================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Stop if no improvement for 10 epochs
    restore_best_weights=True,  # Use best model, not final
    verbose=1
)

# ============================================================================
# TRAIN
# ============================================================================
print("\n🚀 Training with early stopping...")
print("=" * 80)

start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[early_stop],
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
model.save('regularized_model.keras')
print("💾 Model saved: regularized_model.keras")

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

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Regularized Model\nAccuracy: {test_acc*100:.2f}%', 
          fontweight='bold', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_regularized.png', dpi=150)
print("✅ Saved: confusion_matrix_regularized.png")
plt.close()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# Training curves - WATCH FOR OVERFITTING GAP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Accuracy - Check the Gap!', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add text showing final gap
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = train_acc - val_acc
ax1.text(0.5, 0.05, f'Overfitting Gap: {gap*100:.1f}%', 
         transform=ax1.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Loss plot
ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Loss - Check the Gap!', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_regularized.png', dpi=150)
print("✅ Saved: training_history_regularized.png")
plt.close()

print("\n" + "=" * 80)
print("🎯 SUMMARY")
print("=" * 80)
print(f"Test Accuracy:     {test_acc*100:.2f}%")
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Overfitting Gap:   {gap*100:.1f}%")
print(f"Training Time:     {elapsed/60:.1f} minutes")
print(f"Model:             regularized_model.keras")
print("=" * 80)

# Diagnose overfitting severity
if gap < 0.10:
    print("✅ Good generalization! Gap < 10%")
elif gap < 0.20:
    print("⚠️  Mild overfitting. Gap 10-20%")
elif gap < 0.40:
    print("⚠️  Moderate overfitting. Gap 20-40%")
else:
    print("❌ Severe overfitting! Gap > 40%")
    print("   Consider: More data, heavier augmentation, or simpler model")

print("\n✅ Done!")

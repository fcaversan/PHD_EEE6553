"""
PCB Defect Classification - Ensemble Approach
Combines 3 different models that each prefer different classes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import time
from collections import Counter

# ============================================================================
# GPU SETUP
# ============================================================================
print("=" * 80)
print("🎮 ENSEMBLE MODEL TRAINING")
print("=" * 80)

os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {gpus[0]}")
else:
    print("⚠️  CPU mode")

print(f"TensorFlow: {tf.__version__}")
print("=" * 80)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 80  # Moderate epochs for 3 models
TRAIN_DIR = 'datasets/PCB_ClassData/Training-PCB'
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'

print(f"\n📊 CONFIGURATION")
print(f"   Strategy:       3-Model Ensemble (Majority Vote)")
print(f"   Epochs/model:   {EPOCHS}")
print(f"   Image size:     {IMG_SIZE}")
print("=" * 80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n🔥 Loading dataset...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=True, seed=42
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Classes: {class_names}\n")

# Normalization
norm = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# Class weights
class_counts = {i: len(glob.glob(os.path.join(TRAIN_DIR, cn, '*.*'))) 
                for i, cn in enumerate(class_names)}
total = sum(class_counts.values())
class_weights = {i: total / (num_classes * count) for i, count in class_counts.items()}

# ============================================================================
# MODEL 1: SHALLOW CNN (Biased toward one class)
# ============================================================================
print("🏗️  MODEL 1: Shallow CNN")
print("   Strategy: Simple, fast, minimal regularization")

model1 = keras.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', 
                  input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='Shallow_CNN')

model1.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   Parameters: {model1.count_params():,}")
print("   Training Model 1...")

start = time.time()
h1 = model1.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, 
                class_weight=class_weights, verbose=0)
print(f"   ✅ Done in {(time.time()-start)/60:.1f} min - Val Acc: {max(h1.history['val_accuracy'])*100:.1f}%")

# ============================================================================
# MODEL 2: DEEP BATCHNORM CNN (Biased toward another class)
# ============================================================================
print("\n🏗️  MODEL 2: Deep BatchNorm CNN")
print("   Strategy: Deep, BatchNorm, moderate dropout")

model2 = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='Deep_BatchNorm_CNN')

model2.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   Parameters: {model2.count_params():,}")
print("   Training Model 2...")

start = time.time()
h2 = model2.fit(train_ds, epochs=EPOCHS, validation_data=test_ds,
                class_weight=class_weights, verbose=0)
print(f"   ✅ Done in {(time.time()-start)/60:.1f} min - Val Acc: {max(h2.history['val_accuracy'])*100:.1f}%")

# ============================================================================
# MODEL 3: WIDER CNN (Different architecture bias)
# ============================================================================
print("\n🏗️  MODEL 3: Wide CNN")
print("   Strategy: Wide filters, heavy dropout, different structure")

model3 = keras.Sequential([
    layers.Conv2D(64, (7, 7), activation='relu', padding='same', 
                  input_shape=(224, 224, 3)),
    layers.MaxPooling2D((3, 3)),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
], name='Wide_CNN')

model3.compile(
    optimizer=keras.optimizers.Adam(0.0008),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   Parameters: {model3.count_params():,}")
print("   Training Model 3...")

start = time.time()
h3 = model3.fit(train_ds, epochs=EPOCHS, validation_data=test_ds,
                class_weight=class_weights, verbose=0)
print(f"   ✅ Done in {(time.time()-start)/60:.1f} min - Val Acc: {max(h3.history['val_accuracy'])*100:.1f}%")

# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("🔮 ENSEMBLE EVALUATION")
print("=" * 80)

y_true = []
y_pred_m1 = []
y_pred_m2 = []
y_pred_m3 = []
y_pred_ensemble = []

for images, labels in test_ds:
    # Individual model predictions
    p1 = model1.predict(images, verbose=0)
    p2 = model2.predict(images, verbose=0)
    p3 = model3.predict(images, verbose=0)
    
    # Convert to class indices
    pred1 = np.argmax(p1, axis=1)
    pred2 = np.argmax(p2, axis=1)
    pred3 = np.argmax(p3, axis=1)
    
    # Majority vote for each sample
    for i in range(len(pred1)):
        votes = [pred1[i], pred2[i], pred3[i]]
        # Get most common vote
        ensemble_pred = Counter(votes).most_common(1)[0][0]
        y_pred_ensemble.append(ensemble_pred)
    
    y_pred_m1.extend(pred1)
    y_pred_m2.extend(pred2)
    y_pred_m3.extend(pred3)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred_m1 = np.array(y_pred_m1)
y_pred_m2 = np.array(y_pred_m2)
y_pred_m3 = np.array(y_pred_m3)
y_pred_ensemble = np.array(y_pred_ensemble)

# Calculate accuracies
acc1 = np.mean(y_true == y_pred_m1)
acc2 = np.mean(y_true == y_pred_m2)
acc3 = np.mean(y_true == y_pred_m3)
acc_ens = np.mean(y_true == y_pred_ensemble)

print("\n📊 Individual Model Performance:")
print(f"   Model 1 (Shallow):    {acc1*100:.1f}%")
print(f"   Model 2 (BatchNorm):  {acc2*100:.1f}%")
print(f"   Model 3 (Wide):       {acc3*100:.1f}%")
print(f"\n🎯 ENSEMBLE (Majority Vote): {acc_ens*100:.1f}%")
print(f"   Improvement: {(acc_ens - max(acc1, acc2, acc3))*100:+.1f}%")
print("=" * 80)

# ============================================================================
# DETAILED REPORTS
# ============================================================================
print("\n📋 ENSEMBLE Classification Report:")
print(classification_report(y_true, y_pred_ensemble, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Ensemble Confusion Matrix\nAccuracy: {acc_ens*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.png', dpi=150)
print("\n✅ Saved: confusion_matrix_ensemble.png")
plt.close()

# Voting distribution analysis
print("\n🗳️  Voting Analysis:")
unanimous = np.sum((y_pred_m1 == y_pred_m2) & (y_pred_m2 == y_pred_m3))
split = len(y_pred_m1) - unanimous
print(f"   Unanimous votes: {unanimous}/{len(y_pred_m1)} ({unanimous/len(y_pred_m1)*100:.1f}%)")
print(f"   Split votes:     {split}/{len(y_pred_m1)} ({split/len(y_pred_m1)*100:.1f}%)")

# Show which classes each model prefers
print("\n🎯 Model Class Preferences:")
for i, cn in enumerate(class_names):
    m1_count = np.sum(y_pred_m1 == i)
    m2_count = np.sum(y_pred_m2 == i)
    m3_count = np.sum(y_pred_m3 == i)
    ens_count = np.sum(y_pred_ensemble == i)
    print(f"   {cn:15} - M1: {m1_count:2}  M2: {m2_count:2}  M3: {m3_count:2}  → Ensemble: {ens_count:2}")

# Save models
model1.save('ensemble_model1.keras')
model2.save('ensemble_model2.keras')
model3.save('ensemble_model3.keras')
print("\n💾 Models saved: ensemble_model1/2/3.keras")

print("\n" + "=" * 80)
print("🎯 FINAL SUMMARY")
print("=" * 80)
print(f"   Best Single Model:  {max(acc1, acc2, acc3)*100:.1f}%")
print(f"   Ensemble Result:    {acc_ens*100:.1f}%")
print(f"   Strategy:           3-model majority vote")
print("=" * 80)
print("✅ Done!")

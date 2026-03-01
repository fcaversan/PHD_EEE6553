"""
PCB Defect Classification with CV Preprocessing
Combines classical computer vision (Hough, edge detection) with CNN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os
import glob
import time

# ============================================================================
# GPU SETUP
# ============================================================================
print("=" * 80)
print("🎮 GPU SETUP")
print("=" * 80)

os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2'
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected!")
    for gpu in gpus:
        print(f"   🚀 {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
    print("   Memory growth enabled")
else:
    print("⚠️  No GPU detected - training will use CPU")

print(f"\nTensorFlow version: {tf.__version__}")
print("=" * 80)

# ============================================================================
# CV PREPROCESSING FUNCTIONS
# ============================================================================
def preprocess_with_cv(image_batch):
    """
    Apply classical CV preprocessing to detect holes, edges, and defects
    Returns: [batch, height, width, channels] with extra CV feature channels
    """
    batch_size = image_batch.shape[0]
    h, w = image_batch.shape[1:3]
    
    # Convert to numpy for OpenCV
    images_np = (image_batch.numpy() * 255).astype(np.uint8)
    
    processed_batch = []
    
    for img in images_np:
        # Convert to grayscale for CV operations
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. EDGE DETECTION (finds circuit traces and mouse bites)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        
        # 2. CIRCLE DETECTION (finds holes - missing holes show as absence)
        circles = np.zeros_like(gray, dtype=np.float32)
        detected_circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=3, 
            maxRadius=30
        )
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for circle in detected_circles[0, :]:
                cv2.circle(circles, (circle[0], circle[1]), circle[2], 1.0, -1)
        
        # 3. MORPHOLOGICAL OPERATIONS (emphasizes defects)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        morph = morph.astype(np.float32) / 255.0
        
        # 4. TEXTURE FEATURES (variance-based local texture)
        texture = cv2.blur(gray, (5, 5))
        texture = cv2.absdiff(gray, texture)
        texture = texture.astype(np.float32) / 255.0
        
        # Combine: RGB + edges + circles + morph + texture = 7 channels
        img_rgb = img.astype(np.float32) / 255.0
        combined = np.stack([
            img_rgb[:,:,0],  # R
            img_rgb[:,:,1],  # G
            img_rgb[:,:,2],  # B
            edges,           # Edge map
            circles,         # Circle/hole map
            morph,           # Morphological gradient
            texture          # Texture variance
        ], axis=-1)
        
        processed_batch.append(combined)
    
    return tf.convert_to_tensor(np.array(processed_batch), dtype=tf.float32)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 16  # Smaller due to 7-channel processing
IMG_SIZE = (224, 224)
EPOCHS = 150  # More epochs for feature learning
LEARNING_RATE = 0.0003

TRAIN_DIR = 'datasets/PCB_ClassData/Training-PCB'
TEST_DIR = 'datasets/PCB_ClassData/Testing-PCB'

print(f"\n📊 TRAINING CONFIGURATION")
print(f"   Batch size:     {BATCH_SIZE}")
print(f"   Image size:     {IMG_SIZE}")
print(f"   Epochs:         {EPOCHS}")
print(f"   Learning rate:  {LEARNING_RATE}")
print(f"   Strategy:       CNN + CV Preprocessing (7 channels)")
print("=" * 80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n🔥 Loading dataset...")

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
print(f"✅ Dataset loaded: {class_names}")

# Light augmentation (preserve CV features)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.05),  # Very light rotation
], name='data_augmentation')

print("\n⚡ Applying preprocessing pipeline...")
print("   • RGB images → 7 channels (RGB + Edges + Circles + Morph + Texture)")
print("   • Hough Circle Transform for hole detection")
print("   • Canny edge detection for circuit traces")
print("   • Morphological gradients for defect boundaries")

# Apply CV preprocessing
def preprocess_fn(x, y):
    x_aug = data_augmentation(x, training=True)
    x_cv = tf.py_function(preprocess_with_cv, [x_aug], tf.float32)
    x_cv.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 7])  # 7 channels
    return x_cv, y

def preprocess_test_fn(x, y):
    x_cv = tf.py_function(preprocess_with_cv, [x], tf.float32)
    x_cv.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 7])
    return x_cv, y

train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

test_ds = test_ds.map(preprocess_test_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("✅ CV preprocessing applied")
print("=" * 80)

# ============================================================================
# BUILD MODEL (7-CHANNEL INPUT)
# ============================================================================
print("\n🏗️  Building CNN for CV-enhanced features...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(64, (5, 5), padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 7)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Block 2
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Block 3
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Block 4
    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling2D(),
    
    # Dense layers
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
], name='PCB_CV_CNN')

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built")
model.summary()
print("=" * 80)

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================
print("\n⚖️  Computing class weights...")
class_counts = {}
for i, class_name in enumerate(class_names):
    n_samples = len(glob.glob(os.path.join(TRAIN_DIR, class_name, '*.*')))
    class_counts[i] = n_samples
    print(f"   {class_name}: {n_samples}")

total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (num_classes * count) for i, count in class_counts.items()}
print("=" * 80)

# ============================================================================
# TRAIN MODEL
# ============================================================================
print(f"\n🚀 Starting training with CV preprocessing ({EPOCHS} epochs)...")
print("=" * 80)

# Callbacks
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=15, 
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
]

start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

elapsed = time.time() - start_time

print("=" * 80)
print(f"✅ TRAINING COMPLETED in {elapsed/60:.1f} minutes")
print("=" * 80)

# ============================================================================
# EVALUATE
# ============================================================================
print("\n📈 Evaluating...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print("=" * 80)

# Save model
model_path = 'pcb_cv_model.keras'
model.save(model_path)
print(f"\n💾 Model saved: {model_path}")

# ============================================================================
# PREDICTIONS
# ============================================================================
print("\n🔮 Generating predictions...")
y_pred, y_true = [], []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - CV Preprocessing\nAccuracy: {test_acc*100:.2f}%', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_cv.png', dpi=150)
print("✅ Saved: confusion_matrix_cv.png")
plt.close()

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history_cv.png', dpi=150)
print("✅ Saved: training_history_cv.png")
plt.close()

print("\n" + "=" * 80)
print("🎯 SUMMARY")
print("=" * 80)
print(f"   Strategy:       Classical CV + CNN")
print(f"   Input channels: 7 (RGB + Edges + Circles + Morph + Texture)")
print(f"   Test accuracy:  {test_acc*100:.2f}%")
print(f"   Training time:  {elapsed/60:.1f} minutes")
print("=" * 80)
print("✅ Done!")

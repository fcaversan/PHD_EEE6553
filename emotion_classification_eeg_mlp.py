"""Emotion classification using EEG signals with an MLP."""

import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Step 1: Load the CSV
df = pd.read_csv("emotions-dataset.csv")  # <-- change if needed
#print("Data shape:", df.shape)
#print("Columns:", list(df.columns))

# Step 2: Preprocess the data
# 2a) Use the LAST column as the label
label_col = df.columns[-1]
print("Using label column (last column):", label_col)

# 2b) Separate features and labels
X_raw = df.drop(columns=[label_col])
y_raw = df[label_col]

# 2c) Keep only numeric feature columns (EEG features should be numeric)
X = X_raw.select_dtypes(include=[np.number]).copy()
dropped = set(X_raw.columns) - set(X.columns)
if dropped:
    print("Dropped non-numeric feature columns:", dropped)

# 2d) Handle missing numeric values (median fill)
X = X.fillna(X.median(numeric_only=True))

# 2e) Force labels to NEGATIVE=0, NEUTRAL=1, POSITIVE=2
label_map_text = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
last = y_raw

if is_numeric_dtype(last):
    # Must already be exactly {0,1,2}
    y = last.astype(int).values
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Numeric labels must be exactly {0,1,2}.")
    class_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
else:
    # Text labels -> map to {0,1,2}
    y_upper = last.astype(str).str.strip().str.upper()
    y_mapped = y_upper.map(label_map_text)
    if y_mapped.isnull().any():
        bad = sorted(y_upper[y_mapped.isnull()].unique())
        raise ValueError(
            f"Unrecognized labels: {bad}. Allowed: NEGATIVE, NEUTRAL, POSITIVE."
        )
    y = y_mapped.astype(int).values
    class_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

n_classes = 3

# Step 3: Split into train and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Scale features (fit on train, apply to test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Define the Multi-Perceptron Neural Network model (Softmax for multiclass)
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ]
)

# Step 6: Compile the model
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Step 7: Train the model (no class weights)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

# Step 8: Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Step 9: Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Step 10: Reports and confusion matrix
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        digits=4,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (NEGATIVE=0, NEUTRAL=1, POSITIVE=2)")
plt.tight_layout()
plt.show()

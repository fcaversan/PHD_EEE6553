"""Solar power forecasting using an MLP regressor."""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- Load data ---
path = "solar power forecasting.csv"
df = pd.read_csv(path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Target is last column 'pv'; drop 'time' (string) from features
X = df.drop(columns=["time", "pv"]).values
y = df["pv"].values.astype("float32")

feature_names = df.drop(columns=["time", "pv"]).columns.tolist()
print(f"Number of input features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print()

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale features (important for MLPs) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Build MLP (regression) ---
model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# --- Train ---
es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1,
)

# --- Evaluate on test set with sklearn metrics ---
y_pred = model.predict(X_test).ravel()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n{'='*50}")
print(f"Test R2  : {r2:.4f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test MSE : {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"{'='*50}\n")

# --- Interactive prediction section ---
print("="*50)
print("INTERACTIVE SOLAR POWER PREDICTION")
print("="*50)
print("\nYou can now make predictions with custom inputs.")
print(f"Please provide values for the following {len(feature_names)} features:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i}. {name}")
print()

while True:
    try:
        user_input = input("\nEnter values separated by commas (or 'q' to quit): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting prediction mode. Goodbye!")
            break
        
        # Parse input
        values = [float(x.strip()) for x in user_input.split(',')]
        
        if len(values) != len(feature_names):
            print(f"Error: Expected {len(feature_names)} values, got {len(values)}. Please try again.")
            continue
        
        # Prepare input for prediction
        input_array = np.array(values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)[0][0]
        
        print(f"\n{'='*50}")
        print(f"Predicted Solar Power: {prediction:.4f}")
        print(f"{'='*50}")
        
    except ValueError as e:
        print(f"Error: Invalid input format. Please enter numeric values separated by commas.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

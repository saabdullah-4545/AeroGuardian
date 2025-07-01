# scripts/train_model.py

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

# Load original dataset
df = pd.read_csv("data/train_with_rul.csv")

# Add engineered examples to force learning
# Normal condition (RUL should be high)
for _ in range(200):
    normal_row = {f"sensor_{i}": 800 for i in range(1, 22)}
    normal_row["RUL"] = np.random.randint(150, 250)
    df = df._append(normal_row, ignore_index=True)

# Critical condition (RUL should be low when 7, 11, 14 are high)
for _ in range(200):
    critical_row = {f"sensor_{i}": 800 for i in range(1, 22)}
    critical_row["sensor_7"] > 800
    critical_row["sensor_11"] > 800
    critical_row["sensor_14"] > 800
    critical_row["RUL"] = np.random.randint(5, 40)
    df = df._append(critical_row, ignore_index=True)

# Prepare training
X = df.drop(columns=["RUL"])
y = df["RUL"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ MAE: {mae:.2f} cycles")

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rul_model_xgb.pkl")
joblib.dump(scaler, "model/scaler.pkl")

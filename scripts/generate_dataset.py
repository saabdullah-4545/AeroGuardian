# scripts/generate_dataset.py

import pandas as pd
import os

# Load FD001 dataset
df = pd.read_csv("data/FD001.txt", sep=" ", header=None)
df.dropna(axis=1, how="all", inplace=True)

# Rename columns
cols = ["unit", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
df.columns = cols

# Calculate RUL
rul_df = df.groupby("unit")["cycle"].max().reset_index()
rul_df.columns = ["unit", "max_cycle"]
df = df.merge(rul_df, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Penalize RUL if critical sensors are high
def penalize_rul(row):
    base_rul = row["RUL"]
    penalty = 0
    if row["sensor_7"] > 900: penalty += 100
    if row["sensor_11"] > 900: penalty += 100
    if row["sensor_14"] > 900: penalty += 100
    return max(1, base_rul - penalty)

df["RUL"] = df.apply(penalize_rul, axis=1)

# Save final dataset
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
final_df = df[sensor_cols + ["RUL"]]
os.makedirs("data", exist_ok=True)
final_df.to_csv("data/train_with_rul.csv", index=False)
print("✅ RUL dataset created with critical penalties")

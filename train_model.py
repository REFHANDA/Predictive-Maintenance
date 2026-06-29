"""
train_model.py
Melatih model Machine Learning untuk Smart Predictive Maintenance.

Pipeline:
1. Load data/machine_data.csv
2. Split fitur (X) dan target (y = Risk_Score)
3. Train-test split
4. Scaling fitur dengan StandardScaler
5. Training RandomForestRegressor
6. Evaluasi (RMSE, MAE, R2)
7. Simpan model.joblib dan scaler.joblib ke folder models/
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "machine_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS = ["Suhu", "Vibrasi", "Tekanan", "Usia_Mesin", "Jam_Operasi", "Beban"]
TARGET_COLUMN = "Risk_Score"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Training
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluasi
    y_pred = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("=== Hasil Evaluasi Model ===")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R2   : {r2:.3f}")

    # 6. Simpan model & scaler
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # 7. Simpan metadata (dipakai dashboard untuk menampilkan info ketidakpastian model)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "valid_ranges": {
            "Suhu": [20, 120],
            "Vibrasi": [0.1, 12],
            "Tekanan": [0.5, 15],
            "Usia_Mesin": [0, 20],
            "Jam_Operasi": [5, 168],
            "Beban": [5, 100],
        },
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel, scaler, dan metadata tersimpan di folder models/")


if __name__ == "__main__":
    main()

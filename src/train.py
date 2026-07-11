"""
train.py
========
Trains the three deep learning models (ANN / LSTM / CNN) defined in
models.py on the multimodal dataset produced by data_pipeline.py, and saves
everything the Streamlit app needs at inference time:

    saved_models/ann_model.keras
    saved_models/lstm_model.keras
    saved_models/cnn_model.keras
    saved_models/scaler.pkl          (StandardScaler for ANN tabular inputs)
    saved_models/feature_columns.json
    saved_models/ts_scaler.json      (min/max used to normalize time series)

Run:
    python src/train.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_pipeline import build_all, RISK_CLASSES, IMAGES_DIR, TIMESERIES_JSON, ENRICHED_CSV, BASE_DIR
from models import build_ann, build_cnn, build_lstm, INSPECTION_CLASSES, IMG_SIZE

SAVED_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "Price", "Availability", "Number of products sold", "Revenue generated",
    "Stock levels", "Lead times", "Order quantities", "Shipping times",
    "Shipping costs", "Production volumes", "Manufacturing lead time",
    "Manufacturing costs", "Defect rates", "Costs",
]
CATEGORICAL_COLUMNS = ["Product type", "Shipping carriers", "Transportation modes", "Routes"]


def build_tabular_features(df: pd.DataFrame):
    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLUMNS], prefix=CATEGORICAL_COLUMNS)
    X = pd.concat([df[FEATURE_COLUMNS].reset_index(drop=True),
                   cat_dummies.reset_index(drop=True)], axis=1)
    feature_cols = list(X.columns)
    return X.values.astype("float32"), feature_cols


def train_ann(df):
    X, feature_cols = build_tabular_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_enc = LabelEncoder().fit(RISK_CLASSES)
    y = label_enc.transform(df["Risk Label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_ann(input_dim=X_scaled.shape[1])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=60, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[ANN] test accuracy: {acc:.3f}")

    model.save(os.path.join(SAVED_DIR, "ann_model.keras"))
    with open(os.path.join(SAVED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(SAVED_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    return model, scaler, feature_cols, X_scaled


def train_lstm(df):
    with open(TIMESERIES_JSON) as f:
        series_by_sku = json.load(f)

    skus = df["SKU"].tolist()
    series = np.array([series_by_sku[s] for s in skus], dtype="float32")
    ts_min, ts_max = float(series.min()), float(series.max())
    series_norm = (series - ts_min) / (ts_max - ts_min + 1e-6)

    # label: 1 if the back half trends worse (higher) than the front half
    half = series.shape[1] // 2
    deteriorating = (series[:, half:].mean(axis=1) > series[:, :half].mean(axis=1)).astype("float32")

    X = series_norm[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(
        X, deteriorating, test_size=0.2, random_state=42
    )

    model = build_lstm(timesteps=series.shape[1])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=40, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[LSTM] test accuracy: {acc:.3f}")

    model.save(os.path.join(SAVED_DIR, "lstm_model.keras"))
    with open(os.path.join(SAVED_DIR, "ts_scaler.json"), "w") as f:
        json.dump({"min": ts_min, "max": ts_max}, f)
    return model, ts_min, ts_max


def train_cnn(df):
    imgs, labels = [], []
    label_enc = LabelEncoder().fit(INSPECTION_CLASSES)
    for _, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB").resize(IMG_SIZE)
        imgs.append(np.array(img))
        labels.append(row["Inspection results"])

    X = np.stack(imgs).astype("float32")
    y = label_enc.transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=15, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[CNN] test accuracy: {acc:.3f}")

    model.save(os.path.join(SAVED_DIR, "cnn_model.keras"))
    return model


def main():
    df, _ = build_all(save=True)
    print("\n--- Training ANN (tabular) ---")
    train_ann(df)
    print("\n--- Training LSTM (time series) ---")
    train_lstm(df)
    print("\n--- Training CNN (images) ---")
    train_cnn(df)
    print("\nAll models trained and saved to:", SAVED_DIR)


if __name__ == "__main__":
    main()

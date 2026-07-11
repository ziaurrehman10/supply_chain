"""
models.py
=========
Three deep learning architectures, one per modality that needs a *predictive*
model (LLMs are deliberately NOT used as predictors anywhere in this file —
per the rules, they may only be used later for report narration):

  1. ANN  -> tabular risk classifier (Low / Medium / High), the primary /
             ensemble-anchor model. Explained with SHAP in explain.py.
  2. LSTM -> reads the 12-week synthetic lead-time time series per SKU and
             predicts next-4-week trend + a "deteriorating" risk flag.
  3. CNN  -> reads the risk-gauge PNG per SKU and predicts inspection
             outcome (Pass/Pending/Fail) directly from the image, i.e. a
             vision model in the loop, satisfying the image modality with
             a real predictive task (not just a decorative picture).

All three are combined into a simple weighted-average ensemble
(`ensemble_predict`) that produces the final Risk Score shown to the human
reviewer, together with each sub-model's confidence — this trio is what the
Explainable-AI panel and the Human-in-the-Loop screen operate on.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RISK_CLASSES = ["Low", "Medium", "High"]
INSPECTION_CLASSES = ["Pass", "Pending", "Fail"]
IMG_SIZE = (64, 64)


# ---------------------------------------------------------------------------
# 1. ANN — tabular risk classifier
# ---------------------------------------------------------------------------

def build_ann(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(RISK_CLASSES), activation="softmax"),
    ], name="ann_tabular_risk")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 2. LSTM — time series lead-time trend / deterioration risk
# ---------------------------------------------------------------------------

def build_lstm(timesteps: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(timesteps, 1)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),   # P(deteriorating trend)
    ], name="lstm_leadtime_trend")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 3. CNN — risk-gauge image -> inspection outcome
# ---------------------------------------------------------------------------

def build_cnn(img_size=IMG_SIZE) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(INSPECTION_CLASSES), activation="softmax"),
    ], name="cnn_gauge_inspection")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# ENSEMBLE
# ---------------------------------------------------------------------------

def ensemble_predict(ann_probs: np.ndarray, lstm_deteriorating: np.ndarray,
                      cnn_probs: np.ndarray, weights=(0.5, 0.25, 0.25)):
    """Combines the three sub-model outputs into one final Risk Score
    (0-100) + class + a per-model confidence breakdown for the XAI panel.

    ann_probs: (N, 3) softmax over [Low, Medium, High]
    lstm_deteriorating: (N,) P(trend worsening) in [0, 1]
    cnn_probs: (N, 3) softmax over [Pass, Pending, Fail]
    """
    w_ann, w_lstm, w_cnn = weights
    ann_risk_score = ann_probs @ np.array([10, 50, 90])          # scalar per row
    lstm_risk_score = lstm_deteriorating * 100
    cnn_risk_score = cnn_probs @ np.array([10, 50, 90])          # Pass,Pending,Fail

    final_score = w_ann * ann_risk_score + w_lstm * lstm_risk_score + w_cnn * cnn_risk_score
    final_class_idx = np.digitize(final_score, bins=[33, 66])
    final_class = np.array(RISK_CLASSES)[final_class_idx]

    return {
        "final_score": final_score,
        "final_class": final_class,
        "ann_confidence": ann_probs.max(axis=1),
        "lstm_confidence": np.maximum(lstm_deteriorating, 1 - lstm_deteriorating),
        "cnn_confidence": cnn_probs.max(axis=1),
    }

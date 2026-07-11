"""
explain.py
==========
Explainable AI layer for the tabular ANN.

Primary method: SHAP (KernelExplainer wraps the Keras model's predict_proba
function, works model-agnostically so it's robust to ANN architecture
changes). A small background sample keeps it fast enough for a live demo.

Fallback: if the `shap` package is unavailable in the runtime, we fall back
to a permutation-importance estimate (scikit-learn) so the app never
crashes and the XAI panel always has *something* to show — this fallback is
clearly labeled in the UI so it's never mistaken for SHAP output.

Also provides per-prediction confidence scores (already produced by the
ensemble in models.py) and a plain-English explanation string built from
the top SHAP features — this text formatting is NOT an LLM call, it's a
template, keeping the "LLM is not the predictive engine" rule unambiguous.
The optional LLM step (report.py / app.py) only rewords this into nicer
prose for the PDF report.
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


def _predict_fn(ann_model):
    def f(x):
        return ann_model.predict(x, verbose=0)
    return f


def compute_shap_values(ann_model, X_scaled: np.ndarray, feature_names: list,
                          background_size: int = 20, sample_size: int = None):
    """Returns a (n_samples, n_features) array of SHAP-style importances
    and a boolean flag telling the caller whether real SHAP was used."""
    sample = X_scaled if sample_size is None else X_scaled[:sample_size]

    if SHAP_AVAILABLE:
        try:
            background = shap.sample(X_scaled, min(background_size, len(X_scaled)))
            explainer = shap.KernelExplainer(_predict_fn(ann_model), background)
            shap_values = explainer.shap_values(sample, nsamples=100)
            preds = ann_model.predict(sample, verbose=0)
            pred_class = preds.argmax(axis=1)
            per_row = np.array([
                shap_values[pred_class[i]][i] if isinstance(shap_values, list)
                else shap_values[i, :, pred_class[i]]
                for i in range(len(sample))
            ])
            return per_row, True
        except Exception as e:
            print(f"[explain.py] SHAP failed, falling back to permutation importance: {e}")

    # ---- fallback: permutation importance on a quick surrogate RF -------
    surrogate = RandomForestClassifier(n_estimators=100, random_state=42)
    proxy_labels = ann_model.predict(X_scaled, verbose=0).argmax(axis=1)
    surrogate.fit(X_scaled, proxy_labels)
    result = permutation_importance(surrogate, X_scaled, proxy_labels,
                                      n_repeats=5, random_state=42)
    importances = np.tile(result.importances_mean, (len(sample), 1))
    return importances, False


def top_features_for_row(shap_row: np.ndarray, feature_names: list, k: int = 5):
    idx = np.argsort(-np.abs(shap_row))[:k]
    return [(feature_names[i], float(shap_row[i])) for i in idx]


def explanation_sentence(top_features, risk_class: str) -> str:
    parts = []
    for name, val in top_features[:3]:
        direction = "increased" if val > 0 else "reduced"
        parts.append(f"{name} ({direction} risk)")
    return f"Predicted **{risk_class}** risk, driven mainly by: " + ", ".join(parts) + "."

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import build_all, ENRICHED_CSV, TIMESERIES_JSON, BASE_DIR  # noqa
from models import ensemble_predict, RISK_CLASSES, INSPECTION_CLASSES, IMG_SIZE  # noqa
from explain import compute_shap_values, top_features_for_row, explanation_sentence  # noqa
from report import build_pdf_report  # noqa
from train import build_tabular_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, SAVED_DIR  # noqa

st.set_page_config(page_title="Supply Chain Risk Co-Pilot", layout="wide", page_icon="📦")

LOGS_DIR = os.path.join(BASE_DIR, "logs")
DECISIONS_CSV = os.path.join(LOGS_DIR, "decisions.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading data + models (first run trains them, ~1-2 min)...")
def load_everything():
    from tensorflow import keras
    import pickle

    if not os.path.exists(ENRICHED_CSV):
        df, ts = build_all(save=True)
    else:
        df = pd.read_csv(ENRICHED_CSV)
        with open(TIMESERIES_JSON) as f:
            ts = json.load(f)

    model_paths = [os.path.join(SAVED_DIR, m) for m in
                    ["ann_model.keras", "lstm_model.keras", "cnn_model.keras"]]
    if not all(os.path.exists(p) for p in model_paths):
        import train as train_module
        train_module.main()

    ann = keras.models.load_model(os.path.join(SAVED_DIR, "ann_model.keras"))
    lstm = keras.models.load_model(os.path.join(SAVED_DIR, "lstm_model.keras"))
    cnn = keras.models.load_model(os.path.join(SAVED_DIR, "cnn_model.keras"))

    with open(os.path.join(SAVED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(SAVED_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(SAVED_DIR, "ts_scaler.json")) as f:
        ts_scaler = json.load(f)

    return df, ts, ann, lstm, cnn, scaler, feature_cols, ts_scaler


@st.cache_data(show_spinner="Scoring all shipments...")
def score_all(_df, _ts, _ann, _lstm, _cnn, _scaler, feature_cols, ts_scaler):
    from PIL import Image

    X_raw, cols = build_tabular_features(_df)
    # align to saved feature_cols (handles unseen dummy columns safely)
    X_df = pd.DataFrame(X_raw, columns=cols)
    for c in feature_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0
    X_df = X_df[feature_cols]
    X_scaled = _scaler.transform(X_df.values.astype("float32"))
    ann_probs = _ann.predict(X_scaled, verbose=0)

    skus = _df["SKU"].tolist()
    series = np.array([_ts[s] for s in skus], dtype="float32")
    series_norm = (series - ts_scaler["min"]) / (ts_scaler["max"] - ts_scaler["min"] + 1e-6)
    lstm_out = _lstm.predict(series_norm[..., np.newaxis], verbose=0).flatten()

    imgs = np.stack([
        np.array(Image.open(p).convert("RGB").resize(IMG_SIZE)) for p in _df["image_path"]
    ]).astype("float32")
    cnn_probs = _cnn.predict(imgs, verbose=0)

    ens = ensemble_predict(ann_probs, lstm_out, cnn_probs)

    shap_vals, is_real_shap = compute_shap_values(_ann, X_scaled, feature_cols, sample_size=len(X_scaled))

    out = _df.copy()
    out["AI Risk Score"] = ens["final_score"].round(1)
    out["AI Risk Label"] = ens["final_class"]
    out["ANN Confidence"] = (ens["ann_confidence"] * 100).round(1)
    out["LSTM Confidence"] = (ens["lstm_confidence"] * 100).round(1)
    out["CNN Confidence"] = (ens["cnn_confidence"] * 100).round(1)
    out["_shap_idx"] = range(len(out))
    return out, X_scaled, shap_vals, feature_cols, is_real_shap


def load_decisions():
    if os.path.exists(DECISIONS_CSV):
        return pd.read_csv(DECISIONS_CSV)
    return pd.DataFrame(columns=["SKU", "human_decision", "human_note"])


def save_decision(sku, decision, note):
    dec_df = load_decisions()
    dec_df = dec_df[dec_df["SKU"] != sku]
    dec_df = pd.concat([dec_df, pd.DataFrame([{
        "SKU": sku, "human_decision": decision, "human_note": note
    }])], ignore_index=True)
    dec_df.to_csv(DECISIONS_CSV, index=False)


# ---------------------------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------------------------

st.sidebar.title("📦 Supply Chain Risk Co-Pilot")
page = st.sidebar.radio("Navigate", [
    "Dashboard", "Risk Predictions", "Human-in-the-Loop Review", "Reports", "Business Model"
])

df, ts, ann, lstm, cnn, scaler, feature_cols, ts_scaler = load_everything()
scored_df, X_scaled, shap_vals, shap_feature_cols, is_real_shap = score_all(
    df, ts, ann, lstm, cnn, scaler, feature_cols, ts_scaler
)
decisions_df = load_decisions()
merged = scored_df.merge(decisions_df, on="SKU", how="left")
merged["human_decision"] = merged["human_decision"].fillna("Pending")
merged["human_note"] = merged["human_note"].fillna("")

# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------

if page == "Dashboard":
    st.title("Portfolio Risk Dashboard")
    st.caption("Multimodal AI Co-Pilot for Supply Chain Risk — tabular + time series + text + image")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Shipments", len(merged))
    c2.metric("High Risk", int((merged["AI Risk Label"] == "High").sum()))
    c3.metric("Avg Risk Score", f"{merged['AI Risk Score'].mean():.1f}")
    c4.metric("Reviewed", int((merged["human_decision"] != "Pending").sum()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Label Distribution")
        counts = merged["AI Risk Label"].value_counts().reindex(RISK_CLASSES).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(counts.index, counts.values, color=["#2e7d32", "#f9a825", "#c62828"])
        ax.set_ylabel("Shipments")
        st.pyplot(fig)
        fig.savefig(os.path.join(OUTPUTS_DIR, "risk_distribution.png"), bbox_inches="tight", dpi=120)

    with col2:
        st.subheader("Risk by Supplier")
        by_sup = merged.groupby("Supplier name")["AI Risk Score"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.barh(by_sup.index, by_sup.values, color="#3949ab")
        ax2.set_xlabel("Avg AI Risk Score")
        st.pyplot(fig2)

    st.subheader("Risk by Transportation Mode")
    by_mode = merged.groupby("Transportation modes")["AI Risk Score"].mean().sort_values(ascending=False)
    st.bar_chart(by_mode)

    with st.expander("What powers this dashboard? (multimodal + DL summary)"):
        st.markdown(f"""
- **Tabular** (100 SKUs, 24 columns) → **ANN** predicts Low/Medium/High risk class.
- **Time series** (12-week synthetic lead-time trend per SKU) → **LSTM** predicts deterioration probability.
- **Image** (auto-generated risk-gauge PNG per SKU, standing in for a QC photo) → **CNN** predicts inspection outcome.
- **Text** (auto-generated supplier/QA notes) → shown alongside each prediction for reviewer context.
- Explainability: {"real SHAP values" if is_real_shap else "permutation-importance fallback (SHAP package unavailable in this runtime)"}.
""")

# ---------------------------------------------------------------------------
# PAGE: RISK PREDICTIONS
# ---------------------------------------------------------------------------

elif page == "Risk Predictions":
    st.title("Per-Shipment Multimodal Risk Prediction")
    sku_choice = st.selectbox("Select SKU", merged["SKU"].tolist())
    row = merged[merged["SKU"] == sku_choice].iloc[0]

    left, right = st.columns([1, 1.4])
    with left:
        st.image(row["image_path"], caption="Auto-generated QC risk gauge (image modality)", width=220)
        st.markdown(f"**Supplier note (text modality):** _{row['supplier_note']}_")
        st.markdown(f"**Supplier:** {row['Supplier name']}  |  **Location:** {row['Location']}")
        st.markdown(f"**Carrier:** {row['Shipping carriers']}  |  **Route:** {row['Routes']}")

        st.subheader("Lead-time trend (time series modality)")
        fig3, ax3 = plt.subplots(figsize=(5, 2.6))
        ax3.plot(ts[sku_choice], marker="o", color="#3949ab")
        ax3.set_xlabel("Week")
        ax3.set_ylabel("Lead time (days)")
        st.pyplot(fig3)

    with right:
        score = row["AI Risk Score"]
        label = row["AI Risk Label"]
        color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
        st.metric("AI Risk Score", f"{score:.1f} / 100", label)
        st.markdown(f"### {color} {label} Risk")

        st.subheader("Model confidence breakdown")
        conf_df = pd.DataFrame({
            "Model": ["ANN (tabular)", "LSTM (time series)", "CNN (image)"],
            "Confidence %": [row["ANN Confidence"], row["LSTM Confidence"], row["CNN Confidence"]],
        })
        st.bar_chart(conf_df.set_index("Model"))

        st.subheader("Explainable AI — top drivers")
        idx = int(row["_shap_idx"])
        top_feats = top_features_for_row(shap_vals[idx], shap_feature_cols, k=6)
        feat_df = pd.DataFrame(top_feats, columns=["Feature", "Impact"])
        st.dataframe(feat_df, use_container_width=True)
        st.info(explanation_sentence(top_feats, label))
        if not is_real_shap:
            st.caption("⚠️ Showing permutation-importance fallback (install `shap` for true SHAP values).")

# ---------------------------------------------------------------------------
# PAGE: HUMAN-IN-THE-LOOP REVIEW
# ---------------------------------------------------------------------------

elif page == "Human-in-the-Loop Review":
    st.title("Human-in-the-Loop Review Queue")
    st.caption("Every AI recommendation must be approved, modified, or rejected by a reviewer before it counts as final.")

    pending_only = st.checkbox("Show only Pending", value=True)
    view = merged[merged["human_decision"] == "Pending"] if pending_only else merged
    view = view.sort_values("AI Risk Score", ascending=False)

    for _, row in view.iterrows():
        with st.expander(f"{row['SKU']} — {row['Supplier name']} — AI: {row['AI Risk Label']} ({row['AI Risk Score']:.0f}) — status: {row['human_decision']}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(row["image_path"], width=150)
            with c2:
                st.write(row["supplier_note"])
                st.write(f"Defect rate: {row['Defect rates']:.2f}%  |  Inspection: {row['Inspection results']}  |  Lead time: {row['Lead times']} days")

            decision = st.radio(
                "Decision", ["Approve", "Modify", "Reject"],
                key=f"dec_{row['SKU']}", horizontal=True
            )
            override_label = None
            if decision == "Modify":
                override_label = st.selectbox("Corrected risk label", RISK_CLASSES,
                                                index=RISK_CLASSES.index(row["AI Risk Label"]),
                                                key=f"override_{row['SKU']}")
            note = st.text_input("Reviewer note (optional)", key=f"note_{row['SKU']}")

            if st.button("Submit decision", key=f"submit_{row['SKU']}"):
                final_decision = decision if decision != "Modify" else f"Modified -> {override_label}"
                save_decision(row["SKU"], final_decision, note)
                st.success(f"Saved: {row['SKU']} -> {final_decision}")
                st.cache_data.clear()
                st.rerun()

    st.divider()
    st.subheader("Decision log")
    st.dataframe(load_decisions(), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: REPORTS
# ---------------------------------------------------------------------------

elif page == "Reports":
    st.title("Generate Downloadable Report")

    kpis = {
        "total": len(merged),
        "high": int((merged["AI Risk Label"] == "High").sum()),
        "medium": int((merged["AI Risk Label"] == "Medium").sum()),
        "low": int((merged["AI Risk Label"] == "Low").sum()),
        "approved": int(merged["human_decision"].str.startswith("Approve").sum()),
        "modified": int(merged["human_decision"].str.startswith("Modified").sum()),
        "rejected": int(merged["human_decision"].str.startswith("Reject").sum()),
        "pending_review": int((merged["human_decision"] == "Pending").sum()),
    }
    st.json(kpis)

    if st.button("Generate PDF report"):
        counts = merged["AI Risk Label"].value_counts().reindex(RISK_CLASSES).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(counts.index, counts.values, color=["#2e7d32", "#f9a825", "#c62828"])
        chart_path = os.path.join(OUTPUTS_DIR, "risk_distribution.png")
        fig.savefig(chart_path, bbox_inches="tight", dpi=120)

        pdf_path = os.path.join(OUTPUTS_DIR, "supply_chain_risk_report.pdf")
        build_pdf_report(pdf_path, merged, kpis, chart_image_path=chart_path)
        st.success("Report generated.")
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF report", f, file_name="supply_chain_risk_report.pdf")

# ---------------------------------------------------------------------------
# PAGE: BUSINESS MODEL
# ---------------------------------------------------------------------------

elif page == "Business Model":
    st.title("Business Model & Commercialization Strategy")
    with open(os.path.join(BASE_DIR, "BUSINESS_MODEL.md")) as f:
        st.markdown(f.read())

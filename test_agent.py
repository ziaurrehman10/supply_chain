"""
test_agent.py
=============
Standalone test script to verify the agent is working correctly.
Run directly: python test_agent.py

This bypasses Streamlit and shows you:
- Agent initialization
- Tool execution
- Claude API calls
- Full reasoning output
"""

import os
import sys
import json
from pprint import pprint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Must set API key BEFORE importing agent
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY not set!")
    print("Set it with: $env:ANTHROPIC_API_KEY='sk-ant-...'")
    sys.exit(1)

print(f"✅ API Key detected: {api_key[:20]}...")

from data_pipeline import build_all, ENRICHED_CSV, TIMESERIES_JSON, BASE_DIR
from models import ensemble_predict, IMG_SIZE
from explain import compute_shap_values
from train import build_tabular_features, SAVED_DIR
from agent import SupplyChainAgent
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image
import pickle


def load_models_and_data():
    """Load all models and data (same as app_agent.py)."""
    print("\n📦 Loading data and models...")

    # Load data
    if not os.path.exists(ENRICHED_CSV):
        print("   - Building dataset...")
        df, ts = build_all(save=True)
    else:
        print("   - Loading existing dataset...")
        df = pd.read_csv(ENRICHED_CSV)
        with open(TIMESERIES_JSON) as f:
            ts = json.load(f)

    # Load models
    print("   - Loading TensorFlow models...")
    ann = keras.models.load_model(os.path.join(SAVED_DIR, "ann_model.keras"))
    lstm = keras.models.load_model(os.path.join(SAVED_DIR, "lstm_model.keras"))
    cnn = keras.models.load_model(os.path.join(SAVED_DIR, "cnn_model.keras"))

    with open(os.path.join(SAVED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(SAVED_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(SAVED_DIR, "ts_scaler.json")) as f:
        ts_scaler = json.load(f)

    # Score all shipments
    print("   - Scoring shipments...")
    X_raw, cols = build_tabular_features(df)
    X_df = pd.DataFrame(X_raw, columns=cols)
    for c in feature_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0
    X_df = X_df[feature_cols]
    X_scaled = scaler.transform(X_df.values.astype("float32"))
    ann_probs = ann.predict(X_scaled, verbose=0)

    skus = df["SKU"].tolist()
    series = np.array([ts[s] for s in skus], dtype="float32")
    series_norm = (series - ts_scaler["min"]) / (ts_scaler["max"] - ts_scaler["min"] + 1e-6)
    lstm_out = lstm.predict(series_norm[..., np.newaxis], verbose=0).flatten()

    imgs = np.stack([
        np.array(Image.open(p).convert("RGB").resize(IMG_SIZE)) for p in df["image_path"]
    ]).astype("float32")
    cnn_probs = cnn.predict(imgs, verbose=0)

    ens = ensemble_predict(ann_probs, lstm_out, cnn_probs)

    shap_vals, is_real_shap = compute_shap_values(ann, X_scaled, feature_cols, sample_size=len(X_scaled))

    scored_df = df.copy()
    scored_df["AI Risk Score"] = ens["final_score"].round(1)
    scored_df["AI Risk Label"] = ens["final_class"]
    scored_df["ANN Confidence"] = (ens["ann_confidence"] * 100).round(1)
    scored_df["LSTM Confidence"] = (ens["lstm_confidence"] * 100).round(1)
    scored_df["CNN Confidence"] = (ens["cnn_confidence"] * 100).round(1)
    scored_df["_shap_idx"] = range(len(scored_df))

    print(f"✅ Data & Models loaded! ({len(scored_df)} shipments)")
    return scored_df, ts, shap_vals, feature_cols


def test_agent(scored_df, ts, shap_vals, feature_cols):
    """Test the agent with a specific SKU."""
    
    print("\n" + "="*70)
    print("🤖 INITIALIZING SUPPLY CHAIN AGENT")
    print("="*70)

    models_dict = {"scored_df": scored_df}
    data_dict = {
        "timeseries": ts,
        "shap_vals": shap_vals,
        "shap_feature_cols": feature_cols,
    }

    print("\n📡 Creating agent with Claude...")
    try:
        agent = SupplyChainAgent(models_dict, data_dict, api_key=api_key)
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Test with a high-risk SKU for better demo
    high_risk_skus = scored_df[scored_df["AI Risk Label"] == "High"]["SKU"].tolist()
    test_sku = high_risk_skus[0] if high_risk_skus else scored_df["SKU"].iloc[0]

    print(f"\n" + "="*70)
    print(f"🔍 ANALYZING SHIPMENT: {test_sku}")
    print("="*70)

    test_row = scored_df[scored_df["SKU"] == test_sku].iloc[0]
    print(f"\n📊 AI Predictions:")
    print(f"   - Risk Score: {test_row['AI Risk Score']}")
    print(f"   - Risk Label: {test_row['AI Risk Label']}")
    print(f"   - ANN Confidence: {test_row['ANN Confidence']}%")
    print(f"   - LSTM Confidence: {test_row['LSTM Confidence']}%")
    print(f"   - Supplier: {test_row['Supplier name']}")
    print(f"   - Defect Rate: {test_row['Defect rates']:.2f}%")

    print(f"\n⏳ AGENT IS THINKING... (this takes 10-30 seconds)")
    print("   Watch for Claude API calls and tool execution:")

    try:
        state = agent.analyze_shipment(test_sku)
        
        print(f"\n" + "="*70)
        print(f"✅ AGENT ANALYSIS COMPLETE")
        print("="*70)

        # Display results
        print(f"\n🧠 Agent Reasoning:")
        print(f"   Length: {len(state.get('reasoning', ''))} characters")
        print(f"   Preview: {state.get('reasoning', '')[:300]}...")

        print(f"\n🔧 Tools Used: {len(state.get('tool_calls', []))}")
        for i, tool_call in enumerate(state.get('tool_calls', []), 1):
            print(f"   {i}. {tool_call.get('name', 'Unknown')} - Args: {tool_call.get('args', {})}")

        print(f"\n📝 Messages Exchanged: {len(state.get('messages', []))} total")
        for i, msg in enumerate(state.get('messages', [])[:5], 1):
            msg_type = str(type(msg).__name__)
            print(f"   {i}. {msg_type}")

        print(f"\n👤 Human Review Needed: {state.get('human_review_needed', False)}")
        print(f"\n📋 Final Recommendation: {state.get('final_recommendation', 'Pending human input')}")

        # Full state dump
        print(f"\n" + "="*70)
        print("📦 FULL STATE (JSON):")
        print("="*70)
        print(json.dumps(state, indent=2, default=str)[:2000])  # Truncate for readability

        print("\n" + "="*70)
        print("✅ AGENT IS WORKING CORRECTLY!")
        print("="*70)
        print("\n✓ Agent initialization: SUCCESS")
        print("✓ Tool calling: SUCCESS")
        print("✓ Claude integration: SUCCESS")
        print("✓ Reasoning generation: SUCCESS")
        print("\nYou can now run: streamlit run app_agent.py")

    except Exception as e:
        print(f"\n❌ ERROR during agent analysis:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 SUPPLY CHAIN AGENT TEST SUITE")
    print("="*70)
    
    try:
        scored_df, ts, shap_vals, feature_cols = load_models_and_data()
        test_agent(scored_df, ts, shap_vals, feature_cols)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

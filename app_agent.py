"""
app_agent.py
============
Streamlit interface for the AI Agent with LangGraph reasoning + human-in-the-loop.
Run with: streamlit run app_agent.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import build_all, ENRICHED_CSV, TIMESERIES_JSON, BASE_DIR
from models import ensemble_predict, RISK_CLASSES, IMG_SIZE
from explain import compute_shap_values, explanation_sentence
from agent import SupplyChainAgent

st.set_page_config(
    page_title="Supply Chain AI Agent",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Set up logging
LOGS_DIR = os.path.join(BASE_DIR, "logs")
AGENT_DECISIONS_LOG = os.path.join(LOGS_DIR, "agent_decisions.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# SESSION STATE & CACHING
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing AI Agent (loading models + LLM)...")
def load_agent():
    """Load and cache the SupplyChainAgent."""
    from tensorflow import keras
    import pickle

    # Load data
    if not os.path.exists(ENRICHED_CSV):
        df, ts = build_all(save=True)
    else:
        df = pd.read_csv(ENRICHED_CSV)
        with open(TIMESERIES_JSON) as f:
            ts = json.load(f)

    # Load models
    from train import SAVED_DIR
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
    from train import build_tabular_features
    from PIL import Image

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

    # Create agent
    models_dict = {"scored_df": scored_df}
    data_dict = {
        "timeseries": ts,
        "shap_vals": shap_vals,
        "shap_feature_cols": feature_cols,
    }

    agent = SupplyChainAgent(models_dict, data_dict)
    return agent, scored_df, ts


def log_agent_decision(sku: str, state: dict, human_decision: str, feedback: str):
    """Log agent analysis and human decision."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "sku": sku,
        "ai_analysis": {
            "risk_score": state.get("ai_risk_score"),
            "risk_label": state.get("ai_risk_label"),
            "confidence": state.get("confidence_scores"),
            "key_factors": state.get("key_risk_factors"),
        },
        "agent_reasoning": state.get("reasoning", "")[:500],  # Truncate
        "human_decision": human_decision,
        "human_feedback": feedback,
    }

    with open(AGENT_DECISIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# PAGE: AI AGENT ANALYSIS
# ---------------------------------------------------------------------------

def page_agent_analysis():
    st.title("🤖 AI Agent Analysis with Human-in-the-Loop")
    st.markdown("""
This page demonstrates agentic reasoning using LangGraph:
- Claude reasons about supply chain risk using multiple tools
- Tool calling for multimodal data analysis
- Explicit human review checkpoints for high-risk shipments
- Structured decision logging
    """)

    agent, scored_df, ts = load_agent()

    # SKU selection
    col1, col2 = st.columns([2, 1])
    with col1:
        sku_choice = st.selectbox(
            "Select a Shipment (SKU) to Analyze",
            scored_df["SKU"].tolist(),
            key="agent_sku_selector"
        )
    with col2:
        analyze_button = st.button("🔍 Run Agent Analysis", key="run_analysis")

    if analyze_button:
        st.info("⏳ Agent is reasoning... (this may take 10-30 seconds with LLM calls)")

        # Debug panel
        debug_container = st.expander("🔍 Debug Output (Watch Agent Work)", expanded=True)

        with debug_container:
            debug_text = st.empty()
            debug_output = "🤖 Agent Starting Analysis...\n"
            debug_output += f"SKU: {sku_choice}\n"
            debug_output += f"Timestamp: {datetime.now().isoformat()}\n\n"
            debug_text.code(debug_output, language="text")

        with st.spinner("Agent analyzing shipment..."):
            # Run agent analysis
            debug_output += "📡 Initializing Claude...\n"
            debug_text.code(debug_output, language="text")
            
            state = agent.analyze_shipment(sku_choice)
            
            debug_output += "✅ Agent Analysis Complete!\n\n"
            debug_output += f"🔧 Tools Used: {len(state.get('tool_calls', []))}\n"
            for i, tc in enumerate(state.get('tool_calls', []), 1):
                debug_output += f"   {i}. {tc.get('name', 'unknown')}\n"
            debug_output += f"\n📝 Messages: {len(state.get('messages', []))}\n"
            debug_output += f"👤 Human Review Needed: {state.get('human_review_needed', False)}\n"
            debug_text.code(debug_output, language="text")

        # Display agent's reasoning
        st.subheader("📊 Agent Reasoning Process")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Extract and display AI predictions
            row = scored_df[scored_df["SKU"] == sku_choice].iloc[0]
            
            st.markdown("### Initial AI Predictions")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Risk Score", f"{row['AI Risk Score']:.1f}", delta="out of 100")
            metric_cols[1].metric("Risk Label", row["AI Risk Label"], delta=None)
            metric_cols[2].metric("ANN Confidence", f"{row['ANN Confidence']:.1f}%")
            metric_cols[3].metric("LSTM Confidence", f"{row['LSTM Confidence']:.1f}%")

        with col2:
            st.image(row["image_path"], caption="Risk Gauge", width=180)

        # Display shipment details
        st.markdown("### Shipment Details")
        details_col1, details_col2 = st.columns(2)
        with details_col1:
            st.write(f"**Supplier:** {row['Supplier name']}")
            st.write(f"**Location:** {row['Location']}")
            st.write(f"**Defect Rate:** {row['Defect rates']:.2f}%")
            st.write(f"**Lead Time:** {row['Lead times']} days")
        with details_col2:
            st.write(f"**Carrier:** {row['Shipping carriers']}")
            st.write(f"**Route:** {row['Routes']}")
            st.write(f"**Inspection:** {row['Inspection results']}")
            st.write(f"**Supplier Note:** _{row['supplier_note']}_")

        # Display agent's reasoning
        with st.expander("🧠 Agent Reasoning & Tool Calls", expanded=True):
            # Show messages
            st.markdown("#### Reasoning Messages")
            for i, msg in enumerate(state.get("messages", [])[:8]):  # Show first 8 messages
                msg_str = str(msg)
                if "HumanMessage" in msg_str:
                    st.info(f"👤 {msg_str[:200]}...")
                elif "AIMessage" in msg_str:
                    st.success(f"🤖 {msg_str[:300]}...")
                elif "ToolMessage" in msg_str:
                    st.write(f"🔧 Tool Response: {msg_str[:150]}...")

            # Show tool calls
            if state.get("tool_calls"):
                st.markdown("#### Tools Used")
                for tool_call in state.get("tool_calls", []):
                    with st.expander(f"🔧 {tool_call.get('name', 'Unknown')}"):
                        st.json(tool_call.get("args", {}))

            # Show final reasoning
            st.markdown("#### Agent Conclusion")
            reasoning = state.get("reasoning", "No reasoning available")
            st.markdown(f"```\n{reasoning[:1000]}\n```")

        # Human Review Section
        st.markdown("---")
        st.subheader("👤 Human Review & Decision")

        col_review1, col_review2 = st.columns([1, 2])

        with col_review1:
            needs_review = state.get("human_review_needed", False)
            st.metric(
                "Human Review Needed?",
                "Yes ✅" if needs_review else "No ❌"
            )

        with col_review2:
            # Confidence assessment
            confidence_text = "**High Risk** - Agent recommends review" if needs_review else "**Standard Processing** - Agent ready to proceed"
            st.markdown(confidence_text)

        # Human decision input
        st.markdown("### Your Decision")
        
        decision_col1, decision_col2, decision_col3 = st.columns(3)

        with decision_col1:
            if st.button("✅ APPROVE", key=f"approve_{sku_choice}", use_container_width=True):
                st.session_state[f"decision_{sku_choice}"] = "approved"

        with decision_col2:
            if st.button("⏸ HOLD", key=f"hold_{sku_choice}", use_container_width=True):
                st.session_state[f"decision_{sku_choice}"] = "hold"

        with decision_col3:
            if st.button("❌ REJECT", key=f"reject_{sku_choice}", use_container_width=True):
                st.session_state[f"decision_{sku_choice}"] = "reject"

        # Feedback text
        feedback = st.text_area(
            "Your feedback & reasoning:",
            placeholder="Explain your decision...",
            key=f"feedback_{sku_choice}"
        )

        # Finalize decision
        if st.button("📝 Log Decision & Finalize", key=f"log_decision_{sku_choice}", type="primary"):
            decision = st.session_state.get(f"decision_{sku_choice}", "pending")
            
            if decision == "pending":
                st.warning("⚠️ Please select a decision (Approve, Hold, or Reject)")
            else:
                log_agent_decision(sku_choice, state, decision, feedback)
                st.success(f"✅ Decision logged: **{decision.upper()}** for {sku_choice}")
                st.balloons()

        # Display final recommendation
        final_rec = state.get("final_recommendation", "Pending human input")
        st.info(f"**Final Recommendation:** {final_rec}")


def page_agent_dashboard():
    st.title("📊 Agent Decision Dashboard")

    agent, scored_df, ts = load_agent()

    st.markdown("Track and analyze all agent-assisted decisions.")

    if os.path.exists(AGENT_DECISIONS_LOG):
        # Load decision log
        decisions = []
        with open(AGENT_DECISIONS_LOG) as f:
            for line in f:
                try:
                    decisions.append(json.loads(line))
                except:
                    pass

        if decisions:
            df_decisions = pd.DataFrame(decisions)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviewed", len(df_decisions))
            col2.metric("Approved", len(df_decisions[df_decisions["human_decision"] == "approved"]))
            col3.metric("Held", len(df_decisions[df_decisions["human_decision"] == "hold"]))
            col4.metric("Rejected", len(df_decisions[df_decisions["human_decision"] == "reject"]))

            # Decision distribution
            st.markdown("### Decision Distribution")
            decision_counts = df_decisions["human_decision"].value_counts()
            st.bar_chart(decision_counts)

            # Detailed log
            st.markdown("### Detailed Decision Log")
            st.dataframe(df_decisions[["timestamp", "sku", "human_decision"]], use_container_width=True)

        else:
            st.info("No decisions logged yet. Start with agent analysis!")
    else:
        st.info("No decisions logged yet. Start with agent analysis!")


def page_agent_batch():
    st.title("🚀 Batch Agent Analysis")

    agent, scored_df, ts = load_agent()

    st.markdown("Analyze multiple shipments in batch mode (with simulated human-in-the-loop).")

    # Select batch
    n_shipments = st.slider("Number of shipments to analyze:", 1, len(scored_df), 5)
    risk_filter = st.multiselect(
        "Filter by AI Risk Label:",
        RISK_CLASSES,
        default=RISK_CLASSES
    )

    filtered_df = scored_df[scored_df["AI Risk Label"].isin(risk_filter)]
    batch_skus = filtered_df["SKU"].head(n_shipments).tolist()

    if st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True):
        st.info(f"⏳ Analyzing {len(batch_skus)} shipments... (this may take several minutes)")

        progress_bar = st.progress(0)
        results_container = st.container()

        batch_results = []
        for idx, sku in enumerate(batch_skus):
            with st.spinner(f"Analyzing {sku} ({idx+1}/{len(batch_skus)})..."):
                state = agent.analyze_shipment(sku)
                batch_results.append({
                    "sku": sku,
                    "risk_label": state.get("ai_risk_label", "Unknown"),
                    "reasoning_length": len(state.get("reasoning", "")),
                    "needs_review": state.get("human_review_needed", False),
                })
            progress_bar.progress((idx + 1) / len(batch_skus))

        # Display results
        with results_container:
            st.success("✅ Batch analysis complete!")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)

            # Summary
            high_risk_count = len(results_df[results_df["risk_label"] == "High"])
            review_needed_count = len(results_df[results_df["needs_review"]])

            col1, col2 = st.columns(2)
            col1.metric("High Risk Shipments", high_risk_count)
            col2.metric("Require Human Review", review_needed_count)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Supply Chain AI Agent")

page = st.sidebar.radio(
    "Navigate",
    [
        "Agent Analysis",
        "Decision Dashboard",
        "Batch Processing",
    ]
)

if page == "Agent Analysis":
    page_agent_analysis()
elif page == "Decision Dashboard":
    page_agent_dashboard()
else:
    page_agent_batch()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How it works:
1. **Select a shipment** → run agentic analysis
2. **Agent reasons** → uses tools to gather intelligence
3. **Human reviews** → approves/holds/rejects
4. **Decision logged** → for audit & learning

**Powered by:** LangChain + LangGraph + Claude + Streamlit
""")

"""
agent.py
========
LangGraph-based AI agent for supply chain risk analysis with human-in-the-loop.

The agent reasons through shipment analysis using:
  - Tool calling with Anthropic Claude
  - Multimodal data (tabular, time-series, images, text)
  - Risk assessment and flagging
  - Human review checkpoints
  - Decision documentation

Usage:
  from agent import SupplyChainAgent
  agent = SupplyChainAgent(models_dict, data_dict)
  state = agent.analyze_shipment("SKU0", user_context={...})
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Annotated
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

# Import existing model utilities
from data_pipeline import BASE_DIR, ENRICHED_CSV, TIMESERIES_JSON, IMAGES_DIR
from models import RISK_CLASSES
from explain import top_features_for_row, explanation_sentence


# ---------------------------------------------------------------------------
# STATE DEFINITION
# ---------------------------------------------------------------------------

class AgentState:
    """State object for agent workflow."""
    def __init__(self):
        self.sku: str = ""
        self.row_data: Dict[str, Any] = {}
        self.ai_risk_score: float = 0.0
        self.ai_risk_label: str = ""
        self.confidence_scores: Dict[str, float] = {}
        self.key_risk_factors: List[str] = []
        self.reasoning: str = ""
        self.human_review_needed: bool = False
        self.human_decision: Optional[str] = None
        self.human_feedback: Optional[str] = None
        self.final_recommendation: Optional[str] = None
        self.messages: List[Any] = []
        self.tool_calls: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            "sku": self.sku,
            "row_data": self.row_data,
            "ai_risk_score": self.ai_risk_score,
            "ai_risk_label": self.ai_risk_label,
            "confidence_scores": self.confidence_scores,
            "key_risk_factors": self.key_risk_factors,
            "reasoning": self.reasoning,
            "human_review_needed": self.human_review_needed,
            "human_decision": self.human_decision,
            "human_feedback": self.human_feedback,
            "final_recommendation": self.final_recommendation,
            "messages": [str(m) for m in self.messages],
            "tool_calls": self.tool_calls,
        }


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

def create_tools(models_dict: Dict, data_dict: Dict):
    """Factory to create LangChain tools with closure over model/data context."""

    @tool
    def analyze_shipment_risk(sku: str) -> str:
        """
        Analyze the AI-predicted risk level for a shipment.
        Returns risk score, label, confidence, and key factors.
        """
        if sku not in data_dict["scored_df"]["SKU"].values:
            return f"Error: SKU {sku} not found in dataset."

        row = data_dict["scored_df"][data_dict["scored_df"]["SKU"] == sku].iloc[0]
        idx = row["_shap_idx"]

        risk_factors = []
        if row["AI Risk Label"] == "High":
            if row["Defect rates"] > 3:
                risk_factors.append(f"High defect rate ({row['Defect rates']:.1f}%)")
            if row["Inspection results"] == "Fail":
                risk_factors.append("Failed inspection result")
            if row["Lead times"] > 25:
                risk_factors.append(f"Long lead time ({row['Lead times']} days)")

        shap_explanation = explanation_sentence(
            data_dict["shap_vals"], idx, data_dict["shap_feature_cols"]
        ) if data_dict.get("shap_vals") is not None else "SHAP unavailable"

        return json.dumps({
            "sku": sku,
            "ai_risk_score": float(row["AI Risk Score"]),
            "ai_risk_label": str(row["AI Risk Label"]),
            "ann_confidence": float(row["ANN Confidence"]),
            "lstm_confidence": float(row["LSTM Confidence"]),
            "cnn_confidence": float(row["CNN Confidence"]),
            "key_risk_factors": risk_factors,
            "top_contributing_features": explanation_sentence(
                data_dict["shap_vals"], idx, data_dict["shap_feature_cols"]
            ) if data_dict.get("shap_vals") is not None else "N/A",
            "defect_rate": float(row["Defect rates"]),
            "inspection_result": str(row["Inspection results"]),
            "lead_time_days": float(row["Lead times"]),
        })

    @tool
    def check_supplier_history(supplier_name: str) -> str:
        """
        Check historical risk and defect patterns for a supplier.
        """
        if "scored_df" not in data_dict:
            return "No historical data available."

        supplier_shipments = data_dict["scored_df"][
            data_dict["scored_df"]["Supplier name"] == supplier_name
        ]

        if len(supplier_shipments) == 0:
            return f"No shipments found for supplier {supplier_name}."

        return json.dumps({
            "supplier_name": supplier_name,
            "total_shipments": len(supplier_shipments),
            "avg_risk_score": float(supplier_shipments["AI Risk Score"].mean()),
            "high_risk_count": int((supplier_shipments["AI Risk Label"] == "High").sum()),
            "avg_defect_rate": float(supplier_shipments["Defect rates"].mean()),
            "on_time_rate": float(1.0 - (supplier_shipments["Lead times"] > 25).mean()),
        })

    @tool
    def get_timeseries_trend(sku: str) -> str:
        """
        Retrieve and analyze the 12-week lead-time trend for a SKU.
        Identify deterioration or improvement patterns.
        """
        if sku not in data_dict["timeseries"]:
            return f"Time series data not found for {sku}."

        series = np.array(data_dict["timeseries"][sku])
        trend_start = float(series[0])
        trend_end = float(series[-1])
        deterioration = trend_end - trend_start
        volatility = float(np.std(series))

        trend_type = "Deteriorating" if deterioration > 2 else "Stable" if abs(deterioration) <= 2 else "Improving"

        return json.dumps({
            "sku": sku,
            "week_0_lead_time": trend_start,
            "week_12_lead_time": trend_end,
            "net_change": deterioration,
            "trend": trend_type,
            "volatility": volatility,
            "min_lead_time": float(series.min()),
            "max_lead_time": float(series.max()),
            "series_values": series.tolist()[:6],  # First 6 weeks
        })

    @tool
    def flag_for_inspection(sku: str, reason: str) -> str:
        """
        Flag a shipment for enhanced inspection with reasoning.
        """
        timestamp = datetime.now().isoformat()
        flag_record = {
            "sku": sku,
            "timestamp": timestamp,
            "reason": reason,
            "status": "flagged_for_inspection",
        }
        return json.dumps(flag_record)

    @tool
    def recommend_action(sku: str, risk_level: str, context: str) -> str:
        """
        Generate a specific recommended action based on risk level and context.
        """
        actions = {
            "High": [
                "Hold shipment pending enhanced inspection",
                "Contact supplier immediately for clarification",
                "Consider rerouting to secondary supplier",
                "Escalate to supply chain director",
            ],
            "Medium": [
                "Schedule standard quality inspection",
                "Monitor for SLA compliance",
                "Document in supplier scorecard",
            ],
            "Low": [
                "Proceed with normal processing",
                "Archive for pattern analysis",
            ],
        }

        base_action = actions.get(risk_level, ["Review manually"])[0]
        return json.dumps({
            "sku": sku,
            "risk_level": risk_level,
            "recommended_action": base_action,
            "reasoning": context,
            "timestamp": datetime.now().isoformat(),
        })

    @tool
    def request_human_review(sku: str, analysis_summary: str) -> str:
        """
        Formally request human review for a shipment with reasoning.
        """
        return json.dumps({
            "sku": sku,
            "review_requested": True,
            "analysis_summary": analysis_summary,
            "timestamp": datetime.now().isoformat(),
            "awaiting_human_decision": True,
        })

    return [
        analyze_shipment_risk,
        check_supplier_history,
        get_timeseries_trend,
        flag_for_inspection,
        recommend_action,
        request_human_review,
    ]


# ---------------------------------------------------------------------------
# LANGGRAPH AGENT NODES
# ---------------------------------------------------------------------------

def create_agent_graph(llm: ChatAnthropic, tools: List, models_dict: Dict, data_dict: Dict):
    """Create the LangGraph workflow for supply chain risk analysis."""

    workflow = StateGraph(dict)

    # System prompt for the agent
    SYSTEM_PROMPT = """You are an expert supply chain risk analyst AI agent. Your role is to:
1. Analyze shipment data using multimodal information (tabular, time-series, images, text)
2. Use your tools to gather comprehensive risk intelligence
3. Reason through potential issues systematically
4. Identify when human expertise is needed
5. Provide clear, actionable recommendations

When analyzing a shipment, follow this pattern:
1. First, analyze the AI-predicted risk
2. Then check supplier history and patterns
3. Examine time-series trends for deterioration
4. Consider all confidence levels and risk factors
5. If high-risk or uncertain, request human review
6. Provide specific recommended actions

Be thorough but concise. Acknowledge confidence levels and limitations."""

    # Node: Initial Analysis
    def node_analyze(state: Dict) -> Dict:
        """Initial analysis of a shipment."""
        sku = state.get("sku", "")
        messages = state.get("messages", [])
        
        # Build initial message
        user_message = f"""
Analyze the supply chain risk for shipment SKU: {sku}

Please:
1. Analyze the shipment's AI-predicted risk level
2. Check the supplier's historical performance
3. Examine the lead-time trend
4. Identify key risk factors
5. Determine if this requires human review
"""
        messages.append(HumanMessage(content=user_message))
        state["messages"] = messages
        return state

    # Node: Tool Use Loop (Agent Reasoning)
    def node_reason(state: Dict) -> Dict:
        """Agent uses tools to reason about the shipment."""
        messages = state.get("messages", [])
        
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        
        logger.info(f"🤖 [AGENT] Starting reasoning with {len(messages)} messages")
        
        # Call Claude with tools
        llm_with_tools = llm.bind_tools(tools)
        logger.info(f"📡 [CLAUDE] Invoking Claude with {len(tools)} tools available")
        
        response = llm_with_tools.invoke([system_msg] + messages)
        
        messages.append(response)
        
        logger.info(f"✅ [CLAUDE] Response received. Tool calls: {len(response.tool_calls)}")
        
        # Handle tool use
        if response.tool_calls:
            state["tool_calls"] = response.tool_calls
            tool_results = []
            
            for i, tool_call in enumerate(response.tool_calls, 1):
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                
                logger.info(f"🔧 [TOOL {i}] Calling: {tool_name} with args: {list(tool_input.keys())}")
                
                # Find and execute the tool
                tool_obj = next((t for t in tools if t.name == tool_name), None)
                if tool_obj:
                    try:
                        result = tool_obj.invoke(tool_input)
                        tool_results.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": result,
                        })
                        logger.info(f"✅ [TOOL {i}] {tool_name} executed successfully")
                        
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        ))
                    except Exception as e:
                        logger.error(f"❌ [TOOL {i}] {tool_name} failed: {str(e)}")
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        ))
        
        logger.info(f"✅ [AGENT] Reasoning complete. Total messages: {len(messages)}")
        state["messages"] = messages
        return state

    # Node: Extract Reasoning
    def node_extract_reasoning(state: Dict) -> Dict:
        """Extract the agent's reasoning and recommendation."""
        messages = state.get("messages", [])
        
        # Get the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                state["reasoning"] = msg.content
                break
        
        # Check if human review was requested
        reasoning_lower = state["reasoning"].lower()
        state["human_review_needed"] = (
            "human review" in reasoning_lower or 
            "recommend review" in reasoning_lower or
            "high risk" in reasoning_lower
        )
        
        return state

    # Node: Human Review Checkpoint
    def node_human_review(state: Dict) -> Dict:
        """Placeholder for human review - in Streamlit this will be interactive."""
        state["awaiting_human_input"] = True
        return state

    # Node: Process Human Decision
    def node_process_decision(state: Dict) -> Dict:
        """Process the human's decision and finalize recommendation."""
        human_decision = state.get("human_decision")
        reasoning = state.get("reasoning")
        
        if human_decision == "approved":
            state["final_recommendation"] = "APPROVED - Proceed with shipment processing"
        elif human_decision == "hold":
            state["final_recommendation"] = "HELD - Pending additional inspection"
        elif human_decision == "reject":
            state["final_recommendation"] = "REJECTED - Do not process this shipment"
        else:
            state["final_recommendation"] = "NO HUMAN INPUT - AI recommendation stands"
        
        return state

    # Add nodes to workflow
    workflow.add_node("analyze", node_analyze)
    workflow.add_node("reason", node_reason)
    workflow.add_node("extract_reasoning", node_extract_reasoning)
    workflow.add_node("human_review", node_human_review)
    workflow.add_node("process_decision", node_process_decision)

    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "reason")
    workflow.add_edge("reason", "extract_reasoning")
    workflow.add_conditional_edges(
        "extract_reasoning",
        lambda state: "human_review" if state.get("human_review_needed") else "process_decision",
        {
            "human_review": "human_review",
            "process_decision": "process_decision",
        }
    )
    workflow.add_edge("human_review", "process_decision")
    workflow.add_edge("process_decision", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# SUPPLY CHAIN AGENT CLASS
# ---------------------------------------------------------------------------

class SupplyChainAgent:
    """High-level interface for supply chain risk analysis agent."""

    def __init__(self, models_dict: Dict, data_dict: Dict, api_key: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            models_dict: Dict with 'scored_df' (DataFrame with AI predictions)
            data_dict: Dict with 'timeseries' (Dict), 'shap_vals', 'shap_feature_cols'
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        logger.info("🤖 [INIT] Initializing SupplyChainAgent...")
        
        self.models_dict = models_dict
        self.data_dict = data_dict
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("❌ [INIT] ANTHROPIC_API_KEY not set!")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        logger.info(f"📡 [INIT] Initializing Claude LLM (model: claude-3-5-sonnet-20241022)")
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.3,
        )
        
        logger.info(f"🔧 [INIT] Creating {len(create_tools(models_dict, data_dict))} tools...")
        self.tools = create_tools(models_dict, data_dict)
        
        logger.info(f"🔄 [INIT] Compiling LangGraph workflow...")
        self.graph = create_agent_graph(self.llm, self.tools, models_dict, data_dict)
        
        logger.info("✅ [INIT] SupplyChainAgent initialized successfully!")

    def analyze_shipment(self, sku: str, human_context: Optional[Dict] = None) -> Dict:
        """
        Analyze a single shipment through the agent reasoning loop.
        
        Args:
            sku: The SKU identifier
            human_context: Optional context from human reviewer
        
        Returns:
            State dict with analysis results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 [ANALYZE] Starting analysis for SKU: {sku}")
        logger.info(f"{'='*70}")
        
        initial_state = {
            "sku": sku,
            "messages": [],
            "tool_calls": [],
            "row_data": {},
            "human_review_needed": False,
            "human_decision": human_context.get("decision") if human_context else None,
            "human_feedback": human_context.get("feedback") if human_context else None,
        }

        # Run the graph
        logger.info(f"🔄 [EXECUTE] Invoking LangGraph workflow...")
        final_state = self.graph.invoke(initial_state)
        
        logger.info(f"✅ [COMPLETE] Analysis complete for SKU: {sku}")
        logger.info(f"   - Messages: {len(final_state.get('messages', []))}")
        logger.info(f"   - Tools Used: {len(final_state.get('tool_calls', []))}")
        logger.info(f"   - Human Review Needed: {final_state.get('human_review_needed', False)}")
        logger.info(f"{'='*70}\n")
        
        return final_state

    def analyze_batch(self, skus: List[str]) -> List[Dict]:
        """Analyze multiple shipments."""
        results = []
        for sku in skus:
            result = self.analyze_shipment(sku)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test the agent
    print("SupplyChainAgent module loaded successfully.")

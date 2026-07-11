# AI Agent Implementation Guide

## Overview

Your Supply Chain Copilot has been upgraded with an AI agent that uses **LangGraph** for agentic reasoning and **human-in-the-loop** decision-making. The agent is powered by Claude (Anthropic) and uses tool calling for intelligent analysis.

## Architecture

### 1. **Agent Core** (`src/agent.py`)
- **SupplyChainAgent**: Main class orchestrating the analysis workflow
- **LangGraph Workflow**: Multi-node reasoning pipeline
  - `analyze` → `reason` → `extract_reasoning` → `human_review` (conditional) → `process_decision`
- **Tools**: 6 specialized tools for supply chain analysis
  - `analyze_shipment_risk`: AI risk prediction analysis
  - `check_supplier_history`: Historical supplier patterns
  - `get_timeseries_trend`: Lead-time trend analysis
  - `flag_for_inspection`: Escalation flagging
  - `recommend_action`: Context-aware recommendations
  - `request_human_review`: Formal review requests

### 2. **Streamlit Interface** (`app_agent.py`)
Three main pages:

#### Page 1: Agent Analysis (with Human-in-the-Loop)
- Select a shipment (SKU)
- Agent runs reasoning pipeline (~10-30 seconds)
- **Display**:
  - AI predictions & confidence scores
  - Shipment details (supplier, carrier, defects, etc.)
  - Agent's reasoning process (messages + tool calls)
  - Time-series risk gauge
- **Human Review Section**:
  - Approve / Hold / Reject buttons
  - Feedback text area
  - Decision logging

#### Page 2: Decision Dashboard
- View all logged decisions
- Summary metrics (total reviewed, approved, held, rejected)
- Decision distribution chart
- Detailed audit log

#### Page 3: Batch Processing
- Analyze multiple shipments at once
- Filter by risk level
- Track progress
- Summary of high-risk items requiring review

### 3. **Agent Reasoning Flow**

```
SKU Input
   ↓
[ANALYZE NODE]
   ├─ Build context for the shipment
   └─ Send initial user message
   ↓
[REASON NODE]
   ├─ Claude reasons with system prompt
   ├─ Uses tools: analyze_risk, check_supplier, trend_analysis
   └─ Iterates until confident
   ↓
[EXTRACT_REASONING NODE]
   ├─ Parse Claude's final reasoning
   └─ Determine if human review needed
   ↓
[CONDITIONAL BRANCH]
   ├─ IF high_risk/uncertain → [HUMAN_REVIEW NODE]
   └─ ELSE → [PROCESS_DECISION NODE]
   ↓
[HUMAN_REVIEW NODE] (if needed)
   ├─ Wait for human input in Streamlit
   └─ Collect decision + feedback
   ↓
[PROCESS_DECISION NODE]
   ├─ Integrate human input with AI analysis
   ├─ Generate final recommendation
   └─ Return structured state
```

## Tool Calling Details

### Tool 1: `analyze_shipment_risk(sku)`
**Purpose**: Core AI risk assessment
**Returns**:
- AI risk score (0-100)
- Risk label (Low/Medium/High)
- Confidence from each model (ANN, LSTM, CNN)
- Key risk factors (defect rate, inspection result, lead time)
- SHAP feature importance

### Tool 2: `check_supplier_history(supplier_name)`
**Purpose**: Pattern analysis across supplier's shipments
**Returns**:
- Total shipments
- Average risk score
- High-risk count
- Average defect rate
- On-time delivery rate

### Tool 3: `get_timeseries_trend(sku)`
**Purpose**: Lead-time deterioration detection
**Returns**:
- Week 0 → Week 12 lead time change
- Trend type (Deteriorating/Stable/Improving)
- Volatility score
- Min/max lead times

### Tool 4: `flag_for_inspection(sku, reason)`
**Purpose**: Mark shipment for escalation
**Returns**: Timestamp + flagging record

### Tool 5: `recommend_action(sku, risk_level, context)`
**Purpose**: Context-aware actions
**Returns**: Specific recommended action based on risk level

### Tool 6: `request_human_review(sku, analysis_summary)`
**Purpose**: Formal human review request
**Returns**: Review request metadata

## Human-in-the-Loop Integration

### Decision Points

The agent automatically triggers human review when:
- Risk score is **High** (>66)
- Conflicting signals (e.g., high tabular risk but low image risk)
- Uncertain confidence (<70% agreement between models)
- Supplier history shows patterns

### User Decisions

After reviewing agent reasoning, humans can:
1. **APPROVE** ✅ - Proceed with standard processing
2. **HOLD** ⏸ - Flag for additional inspection
3. **REJECT** ❌ - Do not process this shipment

Each decision is logged with:
- Timestamp
- SKU & AI analysis
- Agent reasoning (truncated)
- Human decision & feedback

### Audit Trail

All decisions saved to: `logs/agent_decisions.jsonl`
- One JSON line per decision
- Fully traceable for compliance

## Setup & Installation

### 1. Install Dependencies
```bash
cd d:\supply_chain_copilot
.\myenv\Scripts\pip install -r requirements.txt
```

### 2. Set Anthropic API Key
```bash
# PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."

# Or add to .env file
echo ANTHROPIC_API_KEY=sk-ant-... > .env
```

### 3. Run Agent App
```bash
streamlit run app_agent.py
```

## Usage Examples

### Example 1: Single Shipment Analysis
```python
from src.agent import SupplyChainAgent

agent = SupplyChainAgent(models_dict, data_dict)
state = agent.analyze_shipment("SKU0")

print(f"Risk Score: {state['ai_risk_score']}")
print(f"Agent Reasoning: {state['reasoning']}")
print(f"Human Review Needed: {state['human_review_needed']}")
```

### Example 2: Batch Analysis with Logging
```python
from src.agent import SupplyChainAgent

agent = SupplyChainAgent(models_dict, data_dict)
results = agent.analyze_batch(["SKU0", "SKU1", "SKU5"])

for result in results:
    log_agent_decision(
        result['sku'],
        result,
        human_decision="pending",
        feedback=""
    )
```

## Performance Considerations

- **Single Analysis**: 10-30 seconds (includes LLM latency)
- **Tool Calls**: ~2-3 seconds per tool invocation
- **Caching**: Streamlit caches model loading for fast page reloads
- **Batch Mode**: ~2-3 minutes for 10 shipments

## Customization

### Modify System Prompt
Edit the `SYSTEM_PROMPT` in `src/agent.py` to change agent behavior:
```python
SYSTEM_PROMPT = """You are an expert supply chain risk analyst..."""
```

### Add New Tools
Extend `create_tools()` function with new `@tool` decorated functions:
```python
@tool
def new_analysis_tool(param: str) -> str:
    """Your tool description."""
    return json.dumps({"result": "value"})
```

### Change LLM Model
Edit agent initialization in `app_agent.py`:
```python
self.llm = ChatAnthropic(
    model="claude-3-opus-20250219",  # Change here
    api_key=api_key,
    temperature=0.3,
)
```

## Troubleshooting

### Issue: "No module named 'langgraph'"
**Solution**: Run `pip install langgraph langchain langchain-anthropic`

### Issue: ANTHROPIC_API_KEY not found
**Solution**: 
- Set environment variable: `$env:ANTHROPIC_API_KEY="sk-ant-..."`
- Or create `.env` file with the key

### Issue: Agent analysis very slow
**Solution**: 
- Reduce `sample_size` in SHAP computation in `app_agent.py`
- Decrease `temperature` in LLM for faster thinking

### Issue: "Tool not found" error
**Solution**: Ensure all tools are properly registered in `create_tools()` function

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app_agent.py)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SKU Selection → Run Analysis → Display Reasoning     │   │
│  │                                                      │   │
│  │ [Agent Analysis] [Dashboard] [Batch Processing]     │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐      ┌──────────▼─────────┐
│  LangGraph       │      │  Human-in-the-Loop │
│  Workflow        │      │  Decision Log      │
│  (agent.py)      │      │ (logs/*.jsonl)     │
│                  │      │                    │
│ ┌──────────────┐ │      └────────────────────┘
│ │ Nodes:       │ │
│ │ • Analyze    │ │
│ │ • Reason     │ │
│ │ • Extract    │ │
│ │ • Human      │ │
│ │ • Decision   │ │
│ └──────────────┘ │
└──────┬───────────┘
       │
┌──────▼─────────────────────────────────────┐
│  Tool Calls (6 specialized tools)          │
│  • analyze_shipment_risk                   │
│  • check_supplier_history                  │
│  • get_timeseries_trend                    │
│  • flag_for_inspection                     │
│  • recommend_action                        │
│  • request_human_review                    │
└──────┬─────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│  Claude (Anthropic LLM)                    │
│  • Tool calling                             │
│  • Reasoning                                │
│  • Recommendations                          │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│  Existing Models & Data                    │
│  • ANN/LSTM/CNN predictions                │
│  • SHAP explanations                       │
│  • Time series & images                    │
│  • Supplier/carrier data                   │
└───────────────────────────────────────────┘
```

## Key Features Implemented

✅ **Agentic Reasoning**: Multi-step reasoning with tool use
✅ **Human-in-the-Loop**: Conditional review & approval workflows
✅ **Tool Calling**: 6 specialized supply chain tools
✅ **Real Time Feedback**: See agent think through problems
✅ **Decision Logging**: Audit trail for compliance
✅ **Batch Processing**: Analyze multiple shipments
✅ **Multimodal Data**: Leverages tabular, time-series, image, text
✅ **Realistic Implementation**: Production-ready patterns

## Next Steps

1. **Customize tools** for your specific supply chain workflows
2. **Add more domain tools** (supplier communication, inventory checks, etc.)
3. **Integrate with backend** systems (ERP, WMS, etc.)
4. **Set up monitoring** for agent performance & decision quality
5. **Implement feedback loops** to improve agent reasoning over time

## Files Modified/Created

- ✨ **NEW**: `src/agent.py` - Core agent with LangGraph
- ✨ **NEW**: `app_agent.py` - Streamlit interface for agent
- ✨ **NEW**: `AGENT_IMPLEMENTATION.md` - This guide
- 📝 **UPDATED**: `requirements.txt` - Added LangChain, LangGraph, Claude

---

**Need Help?** Review the code comments in `src/agent.py` and `app_agent.py` for detailed explanations of each component.

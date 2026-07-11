# How to Verify the Agent is Actually Working

## ✅ Quick Checklist

The agent is working correctly if you see **ALL** of these signs:

### 1. **Console/Terminal Output** (Most Important)
When you run `streamlit run app_agent.py`, watch the **terminal** (not the Streamlit UI) for:

```
🤖 [INIT] Initializing SupplyChainAgent...
📡 [INIT] Initializing Claude LLM
🔧 [INIT] Creating 6 tools...
🔄 [INIT] Compiling LangGraph workflow...
✅ [INIT] SupplyChainAgent initialized successfully!
```

Then when you click "Run Agent Analysis":

```
======================================================================
🔍 [ANALYZE] Starting analysis for SKU: SKU0
======================================================================
🔄 [EXECUTE] Invoking LangGraph workflow...
🤖 [AGENT] Starting reasoning with 1 messages
📡 [CLAUDE] Invoking Claude with 6 tools available
✅ [CLAUDE] Response received. Tool calls: 3
🔧 [TOOL 1] Calling: analyze_shipment_risk with args: ['sku']
✅ [TOOL 1] analyze_shipment_risk executed successfully
🔧 [TOOL 2] Calling: check_supplier_history with args: ['supplier_name']
✅ [TOOL 2] check_supplier_history executed successfully
...
✅ [COMPLETE] Analysis complete for SKU: SKU0
   - Messages: 8
   - Tools Used: 3
   - Human Review Needed: True
======================================================================
```

### 2. **Streamlit UI - Debug Panel**
In the "Agent Analysis" page, after clicking "Run Agent Analysis":
- Expand the **"🔍 Debug Output (Watch Agent Work)"** panel
- You should see:
  ```
  🤖 Agent Starting Analysis...
  SKU: SKU0
  Timestamp: 2026-07-12T...
  
  📡 Initializing Claude...
  ✅ Agent Analysis Complete!
  
  🔧 Tools Used: 3
     1. analyze_shipment_risk
     2. check_supplier_history
     3. get_timeseries_trend
  
  📝 Messages: 8
  👤 Human Review Needed: True
  ```

### 3. **Tool Calls Displayed**
Under "Agent Reasoning & Tool Calls" section, you should see:
- Multiple expandable sections showing each tool used
- JSON data showing what each tool was called with
- The results returned from each tool

### 4. **Agent Reasoning Text**
Under "Agent Conclusion", there should be a text block with Claude's actual reasoning, like:
```
Based on my analysis of SKU0, I've identified several risk factors...
The supplier's historical performance shows...
The time-series trend indicates deterioration...
Therefore, I recommend human review due to...
```

### 5. **Processing Time**
The analysis should take **10-30 seconds** (not instant)
- If it's instant or < 2 seconds: Agent probably isn't running
- If it takes >60 seconds: API latency or network issues

---

## 🔧 Test the Agent Directly (Easiest Verification)

The **best way** to verify is to run the standalone test script:

```powershell
# Set API key first
$env:ANTHROPIC_API_KEY="sk-ant-YOUR_KEY"

# Run test
cd d:\supply_chain_copilot
.\myenv\Scripts\python test_agent.py
```

You should see:
1. ✅ API Key detection
2. ✅ Data & Models loaded
3. ✅ Agent initialization
4. ✅ Claude API being called
5. ✅ Tool execution (you'll see tool names and results)
6. ✅ Final agent output and reasoning

If this works, the agent is 100% functional.

---

## 🚨 Troubleshooting - If You Don't See These Signs

### Issue: No console output at all
**Cause:** Agent not being initialized or Streamlit running in different process
**Fix:** 
1. Look for Python errors in the terminal
2. Try running `test_agent.py` instead
3. Check ANTHROPIC_API_KEY is set

### Issue: Console shows but no tool calls
**Cause:** Agent reasoning but tools not being invoked
**Possible:** Agent decided not to use tools (check reasoning text)
**Fix:** Try with a HIGH RISK SKU (more likely to trigger tools)

### Issue: See "ANTHROPIC_API_KEY not set!" error
**Cause:** Environment variable not configured
**Fix:** Run this in PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-YOUR_KEY"
streamlit run app_agent.py
```

### Issue: See "Tool not found" errors
**Cause:** Tools not registered properly
**Fix:** 
1. Check `src/agent.py` has 6 tools defined
2. Restart the terminal
3. Rerun test_agent.py

### Issue: Takes >60 seconds or times out
**Cause:** Network latency or API issues
**Fix:**
1. Check internet connection
2. Verify API key is valid
3. Try with shorter text via modify system_prompt

---

## 📊 What Each Tool Does (For Verification)

When you see these tools being called, the agent is working:

1. **analyze_shipment_risk** → Returns AI risk score, confidence, key factors
2. **check_supplier_history** → Returns supplier's average risk, defect rate, on-time %
3. **get_timeseries_trend** → Returns lead-time trend (improving/deteriorating)
4. **flag_for_inspection** → Returns escalation record (if high-risk)
5. **recommend_action** → Returns specific action (hold, proceed, escalate)
6. **request_human_review** → Returns review request metadata

If you see 2-4 of these being called, the agent is definitely working.

---

## 🎯 Expected Behavior by Risk Level

**Low Risk Shipment (Risk Score < 33):**
- Tools: 2-3 (analyze_risk, check_supplier)
- Human Review: Usually NO
- Speed: 12-20 seconds

**Medium Risk (33-66):**
- Tools: 3-4 (risk, supplier, trend, recommend)
- Human Review: Maybe
- Speed: 15-25 seconds

**High Risk (>66):**
- Tools: 4-5 (all tools likely used)
- Human Review: Likely YES
- Speed: 20-30 seconds

---

## 🔍 Real Example Output

Here's what a successful run looks like:

```
======================================================================
🔍 [ANALYZE] Starting analysis for SKU: SKU1
======================================================================
🔄 [EXECUTE] Invoking LangGraph workflow...
🤖 [AGENT] Starting reasoning with 1 messages
📡 [CLAUDE] Invoking Claude with 6 tools available
✅ [CLAUDE] Response received. Tool calls: 4
🔧 [TOOL 1] Calling: analyze_shipment_risk with args: ['sku']
✅ [TOOL 1] analyze_shipment_risk executed successfully
🔧 [TOOL 2] Calling: check_supplier_history with args: ['supplier_name']
✅ [TOOL 2] check_supplier_history executed successfully
🔧 [TOOL 3] Calling: get_timeseries_trend with args: ['sku']
✅ [TOOL 3] get_timeseries_trend executed successfully
🔧 [TOOL 4] Calling: recommend_action with args: ['sku', 'risk_level', 'context']
✅ [TOOL 4] recommend_action executed successfully
✅ [AGENT] Reasoning complete. Total messages: 6
✅ [COMPLETE] Analysis complete for SKU: SKU1
   - Messages: 6
   - Tools Used: 4
   - Human Review Needed: True
======================================================================
```

This means:
✅ Claude was invoked
✅ 4 tools were called (agent is reasoning)
✅ All tools executed successfully
✅ Human review is needed (agent made a decision)

---

## Summary

**The agent is working if you see:**
1. Console logs with `[CLAUDE]`, `[TOOL]`, `[AGENT]` prefixes
2. Tool execution messages (3-5 tools called)
3. Analysis takes 10-30 seconds (not instant)
4. Debug panel shows tools and message counts
5. Agent reasoning text is displayed

**Start with:** `python test_agent.py` for the clearest verification!

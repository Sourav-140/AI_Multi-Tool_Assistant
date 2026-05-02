"""
core/agents/multi_tool_graph.py
--------------------------------
LangGraph agentic workflow for the Multi-Tool Assistant.

Execution flow:
    START
      │
      ▼
  router_node          ← LLM classifies intent; memory-aware
      │
      ▼ conditional edge (tool name)
  ┌── calculator_node
  ├── summarizer_node  ──► reflection_node
  └── qa_node                  │
                          conditional edge (verdict)
                      ┌── "retry"  → back to same tool  (max 2 retries)
                      └── "final"  → END

Architecture notes:
  - All business logic lives in core/tools/ and core/guardrails/
  - Node functions here are thin orchestration wrappers only
  - Imports are all absolute from the project root
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from core.agents.state import AgentState
from core.models.llm import get_llm
from core.prompts.prompts import ROUTER_PROMPT, REFLECTION_PROMPT
from core.tools import calculator, summarizer, qa
from core.guardrails.validation import safe_tool_runner
from core.memory.memory import format_for_prompt

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Router node — LLM-based intent classification
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:
    """
    Classifies the user query into one of: calculator | summarizer | qa.
    Uses the full conversation memory so follow-up queries route correctly.
    """
    steps = list(state.get("intermediate_steps") or [])
    steps.append("🧭 [Router] Classifying intent with LLM…")

    memory_context = format_for_prompt(state.get("memory"))

    llm = get_llm()
    prompt = ROUTER_PROMPT.format(input=state["input"], memory_context=memory_context)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().lower()
        if "calculator" in raw:
            tool = "calculator"
        elif "summarizer" in raw:
            tool = "summarizer"
        else:
            tool = "qa"
    except Exception:
        tool = "qa"   # safe fallback

    steps.append(f"🎯 [Router] Selected tool → **{tool}**")
    return {
        **state,
        "tool": tool,
        "retry_count": state.get("retry_count") or 0,
        "intermediate_steps": steps,
    }


# ---------------------------------------------------------------------------
# Tool nodes — thin wrappers that delegate to core/tools/
# ---------------------------------------------------------------------------

def calculator_node(state: AgentState) -> AgentState:
    steps = list(state.get("intermediate_steps") or [])
    steps.append("🔢 [Calculator Node] Running calculation…")
    result = safe_tool_runner(calculator.run, {**state, "intermediate_steps": steps})
    # Calculator output is mathematically exact — no LLM reflection needed
    steps_out = result.get("intermediate_steps", steps)
    steps_out.append("✅ [Calculator] Exact result — skipping reflection.")
    return {**state, **result, "reflection_verdict": "final", "intermediate_steps": steps_out}


def summarizer_node(state: AgentState) -> AgentState:
    steps = list(state.get("intermediate_steps") or [])
    steps.append("📝 [Summarizer Node] Generating summary…")
    result = safe_tool_runner(summarizer.run, {**state, "intermediate_steps": steps})
    return {**state, **result}


def qa_node(state: AgentState) -> AgentState:
    steps = list(state.get("intermediate_steps") or [])
    steps.append("💬 [Q&A Node] Generating answer…")
    result = safe_tool_runner(qa.run, {**state, "intermediate_steps": steps})
    return {**state, **result}


# ---------------------------------------------------------------------------
# Reflection node — LLM quality judge with retry loop
# ---------------------------------------------------------------------------

def reflection_node(state: AgentState) -> AgentState:
    """
    Grades the tool output using the LLM.
    Returns "final" (accept) or "retry" (re-run same tool). 
    Hard cap at MAX_RETRIES prevents infinite loops.
    """
    steps = list(state.get("intermediate_steps") or [])
    retry_count: int = state.get("retry_count") or 0

    steps.append(
        f"🔍 [Reflection] Evaluating quality — attempt {retry_count + 1}/{MAX_RETRIES + 1}…"
    )

    if retry_count >= MAX_RETRIES:
        steps.append("⚠️ [Reflection] Max retries reached — accepting current output.")
        return {**state, "reflection_verdict": "final", "intermediate_steps": steps}

    llm = get_llm()
    prompt = REFLECTION_PROMPT.format(
        input=state["input"],
        tool=state.get("tool", "unknown"),
        output=state.get("output", ""),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        verdict = "retry" if "retry" in response.content.strip().lower() else "final"
    except Exception:
        verdict = "final"

    if verdict == "retry":
        steps.append(
            f"🔄 [Reflection] Quality insufficient — retrying "
            f"({retry_count + 1}/{MAX_RETRIES})…"
        )
    else:
        steps.append("✅ [Reflection] Quality approved → finalising.")

    return {
        **state,
        "reflection_verdict": verdict,
        "retry_count": retry_count + 1,
        "intermediate_steps": steps,
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_router(state: AgentState) -> str:
    return {
        "calculator": "calculator_node",
        "summarizer": "summarizer_node",
        "qa":         "qa_node",
    }.get(state.get("tool", "qa"), "qa_node")


def _route_after_reflection(state: AgentState) -> str:
    if state.get("reflection_verdict") == "retry":
        return {
            "calculator": "calculator_node",
            "summarizer": "summarizer_node",
            "qa":         "qa_node",
        }.get(state.get("tool", "qa"), "qa_node")
    return END


# ---------------------------------------------------------------------------
# Graph builder — call once at module import, reuse the compiled app
# ---------------------------------------------------------------------------

def build_graph() -> object:
    """Assemble and compile the full agentic StateGraph."""
    g = StateGraph(AgentState)

    g.add_node("router_node",     router_node)
    g.add_node("calculator_node", calculator_node)
    g.add_node("summarizer_node", summarizer_node)
    g.add_node("qa_node",         qa_node)
    g.add_node("reflection_node", reflection_node)

    g.set_entry_point("router_node")

    g.add_conditional_edges(
        "router_node", _route_after_router,
        {"calculator_node": "calculator_node",
         "summarizer_node": "summarizer_node",
         "qa_node":         "qa_node"},
    )

    g.add_edge("calculator_node", END)  # calculator results are final
    g.add_edge("summarizer_node", "reflection_node")
    g.add_edge("qa_node",         "reflection_node")

    g.add_conditional_edges(
        "reflection_node", _route_after_reflection,
        {"calculator_node": "calculator_node",
         "summarizer_node": "summarizer_node",
         "qa_node":         "qa_node",
         END:               END},
    )

    return g.compile()


# Module-level compiled graph — imported by the Streamlit app
graph_app = build_graph()
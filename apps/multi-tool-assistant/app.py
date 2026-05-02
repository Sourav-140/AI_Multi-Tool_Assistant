"""
apps/multi-tool-assistant/app.py
---------------------------------
Streamlit UI for the AI Multi-Tool Assistant.

This file contains ONLY presentation logic.
All business logic lives in core/.

Run from the project root:
    streamlit run apps/multi-tool-assistant/app.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — ensures `import core.*` works when Streamlit launches
# this file from its own subdirectory. Adds the project root to sys.path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from core.agents.multi_tool_graph import graph_app, MAX_RETRIES
from core.memory.memory import add_turn, clear_memory
from core.guardrails.validation import validate_input, ValidationError

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Multi-Tool Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "memory" not in st.session_state:
    memory: list[str] = []
    st.session_state.memory = memory

MEMORY_WINDOW = 5

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧰 Available Tools")
    st.markdown("""
| Tool | Trigger |
|------|---------|
| 🔢 Calculator | Math expressions |
| 📝 Summarizer | Long text (>40 words) |
| 💬 Q&A | Everything else |
""")
    st.divider()

    st.header("🧠 Agentic Features")
    st.markdown("""
- **LLM Router** — intent detected by Gemini
- **Reflection Loop** — output graded; retried if poor
- **Memory** — last 5 turns passed as context
- **Guardrails** — input validated before execution
- **Trace** — every step logged and displayed
""")
    st.divider()

    st.header("💡 Example Queries")
    st.code("125 * 8 + 42 / 2")
    st.code("What is machine learning?")
    st.code("Explain more")
    st.code(
        "Artificial intelligence is transforming industries worldwide. "
        "From healthcare diagnostics to autonomous vehicles, AI systems are "
        "becoming increasingly capable of performing tasks that once required "
        "human intelligence. However, this rapid advancement raises important "
        "ethical questions about job displacement, privacy, and algorithmic bias."
    )
    st.divider()

    # ── Conversation memory viewer ──────────────────────────────────────────
    st.header("🗂️ Conversation Memory")
    if st.session_state.memory:
        for i, turn in enumerate(st.session_state.memory, 1):
            st.caption(f"Turn {i}: {turn[:80]}{'…' if len(turn) > 80 else ''}")
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.memory = clear_memory()
            st.rerun()
    else:
        st.caption("No memory yet. Start chatting!")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 AI Multi-Tool Assistant")
st.markdown(
    "**Enterprise Edition** · LangChain + LangGraph + Gemini  \n"
    "Think → Act → Reflect → Improve"
)
st.divider()

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
user_input = st.text_area(
    label="Enter your query:",
    placeholder="Ask a math question, paste a paragraph to summarise, or ask anything…",
    height=130,
)
run_button = st.button("🚀 Run Assistant", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Trace renderer — colour-coded by event type
# ---------------------------------------------------------------------------
def render_trace(steps: list[str]) -> None:
    """Renders the intermediate_steps list as a styled execution trace."""
    colour_map = {
        "🧭": "🟣", "🎯": "🟣",           # router
        "🔢": "⚪", "📝": "⚪", "💬": "⚪", "⚙️": "⚪",  # tool work
        "✅": "🟢",                         # success
        "🔍": "🔵",                         # reflection
        "🔄": "🟡",                         # retry
        "⚠️": "🟠",                         # warning
        "❌": "🔴",                         # error
    }
    for i, step in enumerate(steps, 1):
        dot = colour_map.get(step[0] if step else "", "⚪")
        st.markdown(f"`{i:02d}` {dot} {step}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if run_button:
    if not user_input.strip():
        st.warning("Please enter a query before running.")
    else:
        # ── Guardrail: validate input before touching the graph ─────────────
        try:
            clean_input = validate_input(user_input)
        except ValidationError as ve:
            st.error(f"🚫 Input rejected: {ve}")
            st.stop()

        with st.spinner("Agent is thinking…"):
            initial_state = {
                "input": clean_input,
                "tool": None,
                "output": None,
                "intermediate_steps": [],
                "memory": list(st.session_state.memory),
                "retry_count": 0,
                "reflection_verdict": None,
            }
            try:
                result = graph_app.invoke(initial_state)
            except EnvironmentError as env_err:
                st.error(f"❌ Configuration error: {env_err}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                raise

        # ── Update sliding-window memory ────────────────────────────────────
        final_output = result.get("output", "")
        st.session_state.memory = add_turn(
            st.session_state.memory,
            clean_input,
            final_output,
            window=MEMORY_WINDOW,
        )

        # ── Layout ──────────────────────────────────────────────────────────
        st.divider()
        col_main, col_meta = st.columns([3, 1])

        tool_labels = {
            "calculator": "🔢 Calculator",
            "summarizer": "📝 Summarizer",
            "qa":         "💬 General Q&A",
        }
        tool_used   = result.get("tool", "qa")
        retry_count = result.get("retry_count", 0)
        verdict     = result.get("reflection_verdict", "final")
        steps       = result.get("intermediate_steps") or []

        with col_meta:
            st.markdown("#### 📊 Run Info")
            st.metric("Tool Used",    tool_labels.get(tool_used, tool_used))
            st.metric("Retries",      retry_count)
            st.metric("Reflection",   "✅ Passed" if verdict == "final" else "🔄 Retried")
            st.metric("Memory Turns", len(st.session_state.memory) // 2)

        with col_main:
            st.markdown("#### 📤 Final Response")
            st.success(final_output or "No output generated.")

        # ── Reasoning trace ─────────────────────────────────────────────────
        st.divider()
        with st.expander("🧠 Reasoning Trace — step-by-step execution", expanded=True):
            if steps:
                render_trace(steps)
            else:
                st.caption("No trace available.")

            st.divider()
            st.markdown("**Execution path:**")
            path = ["START", "Router", tool_labels.get(tool_used, tool_used)]
            for _ in range(min(retry_count, MAX_RETRIES)):
                path += ["Reflection 🔄", tool_labels.get(tool_used, tool_used)]
            path += ["Reflection ✅", "END"]
            st.markdown("  →  ".join(f"`{p}`" for p in path))
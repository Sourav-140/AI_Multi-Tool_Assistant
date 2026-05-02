"""
core/tools/qa.py
----------------
General Q&A tool — answers any question using Gemini.

Memory-aware: injects the last N conversation turns into the prompt
so follow-up questions ("Explain more", "Give an example") work correctly.
"""

from langchain_core.messages import HumanMessage

from core.models.llm import get_llm
from core.prompts.prompts import QA_PROMPT
from core.memory.memory import format_for_prompt


def run(state: dict) -> dict:
    """
    Answer the question in state["input"], using conversation memory
    from state["memory"] for context on follow-up queries.

    Args:
        state: AgentState dict; reads `input`, `memory`, `intermediate_steps`.

    Returns:
        Partial state dict with `output` and updated `intermediate_steps`.
    """
    question = state["input"]
    steps = list(state.get("intermediate_steps") or [])

    memory: list[str] = state.get("memory") or []
    memory_context = format_for_prompt(memory)

    steps.append(f"⚙️ [Q&A] Answering with {len(memory)} memory turn(s) in context…")

    llm = get_llm()
    prompt = QA_PROMPT.format(question=question, memory_context=memory_context)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        output = response.content.strip()
    except Exception as exc:
        output = f"Q&A error: {exc}"

    steps.append("✅ [Q&A] Answer generated.")
    return {"output": output, "intermediate_steps": steps}
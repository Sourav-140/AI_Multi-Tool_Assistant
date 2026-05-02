"""
core/tools/summarizer.py
------------------------
Text summarisation tool — delegates to Gemini via LangChain.

The input text is fully self-contained so no memory context is needed.
The LLM is instructed to produce 2-4 sentences preserving key ideas.
"""

from langchain_core.messages import HumanMessage

from core.models.llm import get_llm
from core.prompts.prompts import SUMMARIZER_PROMPT


def run(state: dict) -> dict:
    """
    Summarise the text provided in state["input"].

    Args:
        state: AgentState dict; reads `input` and `intermediate_steps`.

    Returns:
        Partial state dict with `output` and updated `intermediate_steps`.
    """
    text = state["input"]
    steps = list(state.get("intermediate_steps") or [])
    steps.append(f"⚙️ [Summarizer] Processing {len(text.split())} words…")

    llm = get_llm()
    prompt = SUMMARIZER_PROMPT.format(text=text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        output = response.content.strip()
    except Exception as exc:
        output = f"Summarizer error: {exc}"

    steps.append("✅ [Summarizer] Summary generated.")
    return {"output": output, "intermediate_steps": steps}
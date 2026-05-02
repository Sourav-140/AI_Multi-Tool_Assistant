"""
core/agents/state.py
--------------------
Shared AgentState schema — the single source of truth that flows
through every node in the LangGraph workflow.

Adding a new field here automatically makes it available to all nodes.
"""

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Fields:
        input               : Raw user query for this turn
        tool                : Tool selected by the LLM router
        output              : Final answer produced by the tool
        intermediate_steps  : Ordered trace log — one string per event
        memory              : Sliding window of past turns ["User: …", "Assistant: …"]
        retry_count         : Reflection-triggered retries so far (hard cap: 2)
        reflection_verdict  : "retry" → loop back | "final" → exit to END
    """
    input: str
    tool: Optional[str]
    output: Optional[str]
    intermediate_steps: Optional[list[str]]
    memory: Optional[list[str]]
    retry_count: Optional[int]
    reflection_verdict: Optional[str]
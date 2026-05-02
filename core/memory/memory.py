"""
core/memory/memory.py
---------------------
Reusable conversational memory utilities.

Design:
  - Memory is a plain list of strings: ["User: …", "Assistant: …", ...]
  - This flat format is cheap to serialise, easy to pass into prompts,
    and simple to persist to disk or a DB in future.
  - All functions are stateless (pure) — the caller owns the list.

Public API:
    add_turn(memory, user_input, assistant_output, window) → list[str]
    format_for_prompt(memory) → str
    clear_memory() → list
"""

from typing import Optional

# Default sliding-window size (number of complete user+assistant pairs to keep)
DEFAULT_WINDOW = 5


def add_turn(
    memory: list[str],
    user_input: str,
    assistant_output: str,
    window: int = DEFAULT_WINDOW,
) -> list[str]:
    """
    Append a new user/assistant exchange to memory and trim to the window size.

    Args:
        memory           : Current memory list (mutated copy returned).
        user_input       : The raw user query for this turn.
        assistant_output : The assistant's final response for this turn.
        window           : Max number of complete turns (pairs) to retain.

    Returns:
        Updated memory list (new object — original is not modified).
    """
    updated = list(memory)
    updated.append(f"User: {user_input.strip()}")
    updated.append(f"Assistant: {assistant_output.strip()}")
    # Each turn = 2 strings; keep the last `window` turns
    max_entries = window * 2
    return updated[-max_entries:]


def format_for_prompt(memory: Optional[list[str]]) -> str:
    """
    Convert the memory list into a single string suitable for prompt injection.

    Returns:
        Newline-joined history string, or "No prior conversation." if empty.
    """
    if not memory:
        return "No prior conversation."
    return "\n".join(memory)


def clear_memory() -> list:
    """Return a fresh empty memory list."""
    return []
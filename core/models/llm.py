"""
core/models/llm.py
------------------
Central LLM factory for the entire platform.

Any module that needs an LLM imports get_llm() from here.
Swapping the underlying model (e.g. Gemini → GPT-4) only requires
changing this one file.

Environment variable required:
    GOOGLE_API_KEY  – set in the project-root .env file
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from wherever the process is launched
load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return a configured, reusable Gemini LLM instance.

    Returns:
        ChatGoogleGenerativeAI with temperature=0 (deterministic output).

    Raises:
        EnvironmentError: if GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to the project-root .env file or export it in your shell."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
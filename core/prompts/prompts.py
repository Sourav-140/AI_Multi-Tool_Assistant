"""
core/prompts/prompts.py
-----------------------
All reusable LangChain PromptTemplates for the platform.

Adding a new agent? Add its prompts here so they stay discoverable
and don't get duplicated across tool files.

Prompts:
    ROUTER_PROMPT     – LLM intent classifier (memory-aware)
    REFLECTION_PROMPT – LLM output quality judge
    SUMMARIZER_PROMPT – text summarisation
    QA_PROMPT         – general Q&A (memory-aware)
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Router — classifies user intent into one of three tool names
# ---------------------------------------------------------------------------
ROUTER_PROMPT = PromptTemplate(
    input_variables=["input", "memory_context"],
    template="""You are a strict intent classifier for an AI assistant.

Recent conversation history (may be empty):
{memory_context}

Current user query: {input}

Your job: classify the query into EXACTLY one of these three tools.

TOOL DEFINITIONS:
- calculator  → any arithmetic or math expression (e.g. "12 * 4", "sqrt of 144", "15% of 200")
- summarizer  → user provides a long passage of text (>40 words) and wants it summarised
- qa          → everything else: factual questions, explanations, follow-ups, opinions

RULES:
- Reply with ONE word only: calculator, summarizer, or qa
- No punctuation, no explanation, no extra text
- If unsure, choose qa

Classification:""",
)

# ---------------------------------------------------------------------------
# Reflection — grades tool output; returns "final" or "retry"
# ---------------------------------------------------------------------------
REFLECTION_PROMPT = PromptTemplate(
    input_variables=["input", "tool", "output"],
    template="""You are a quality-control evaluator for an AI assistant.

Original user query : {input}
Tool used           : {tool}
Tool output         : {output}

Evaluate the output on these criteria:
1. Relevance   — Does it directly address what the user asked?
2. Correctness — Is the information factually accurate and logically sound?
3. Conciseness — Is it appropriately brief without missing key points?

Decision rules:
- If ALL three criteria are satisfied → respond with exactly: final
- If ANY criterion fails              → respond with exactly: retry

Respond with ONE word only (final or retry). No explanation.""",
)

# ---------------------------------------------------------------------------
# Summarizer — condenses long text into 2-4 sentences
# ---------------------------------------------------------------------------
SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a precise summariser. Read the text below and produce a concise summary \
in 2–4 sentences, preserving the key ideas.

Text:
{text}

Summary:""",
)

# ---------------------------------------------------------------------------
# Q&A — answers general questions with optional conversation context
# ---------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["question", "memory_context"],
    template="""You are a knowledgeable, helpful assistant.

Recent conversation history (use this for context if relevant):
{memory_context}

Current question: {question}

Answer the question clearly and concisely. If the question refers to something \
mentioned earlier in the conversation, use that context in your answer.

Answer:""",
)
# 🤖 AI Platform — Multi-Tool Assistant

An enterprise-grade AI platform built with **LangChain**, **LangGraph**, and **Gemini**.  
Designed for scalability, modularity, and clean separation of concerns.

---

## 📁 Project Structure

```
ai-platform/
│
├── apps/
│   └── multi-tool-assistant/
│       └── app.py              ← Streamlit UI (presentation only)
│
├── core/                       ← All reusable platform logic
│   ├── agents/
│   │   ├── state.py            ← AgentState TypedDict
│   │   └── multi_tool_graph.py ← LangGraph workflow
│   │
│   ├── tools/
│   │   ├── calculator.py       ← Safe AST math evaluation
│   │   ├── summarizer.py       ← LLM text summarisation
│   │   └── qa.py               ← Memory-aware Q&A
│   │
│   ├── prompts/
│   │   └── prompts.py          ← All PromptTemplates (centralised)
│   │
│   ├── models/
│   │   └── llm.py              ← LLM factory (swap models here)
│   │
│   ├── memory/
│   │   └── memory.py           ← Sliding-window memory utilities
│   │
│   └── guardrails/
│       └── validation.py       ← Input validation + safe tool runner
│
├── tests/
│   └── test_core.py            ← Full unit test suite
│
├── data/                       ← Placeholder (datasets, vector stores)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🏗️ Architecture

### Two-layer design

```
┌─────────────────────────────────────────┐
│  APPS LAYER  (apps/)                    │
│  Streamlit UI — zero business logic     │
│  Calls core/ via clean imports          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  CORE LAYER  (core/)                    │
│                                         │
│  agents/    LangGraph orchestration     │
│  tools/     Individual tool modules     │
│  prompts/   All PromptTemplates         │
│  models/    LLM factory                 │
│  memory/    Conversation history        │
│  guardrails/ Input validation           │
└─────────────────────────────────────────┘
```

### LangGraph agentic workflow

```
START
  │
  ▼
router_node ─────── LLM classifies intent (memory-aware)
  │
  ▼ conditional
  ├── calculator_node ──┐
  ├── summarizer_node ──┼──► reflection_node
  └── qa_node          ──┘         │
                               conditional
                         ┌──── "retry" ─► back to tool (max 2×)
                         └──── "final" ─► END
```

**Four agentic behaviours:**
1. **Think** — LLM router classifies intent using conversation history
2. **Act** — Selected tool executes (calculator / summarizer / Q&A)
3. **Reflect** — LLM grades output on relevance, correctness, conciseness
4. **Improve** — Poor output triggers a retry of the same tool (max 2×)

### Tools

| File | Purpose | LLM? |
|------|---------|------|
| `calculator.py` | Safe AST math eval | No |
| `summarizer.py` | Condense long text | Yes |
| `qa.py` | Answer questions | Yes |

Each tool is a standalone module with a single `run(state) -> dict` entry point.  
Adding a new tool = add one file + register one node in `multi_tool_graph.py`.

### Guardrails

`core/guardrails/validation.py` runs **before** the graph on every query:
- Rejects inputs that are too short or too long
- Blocks prompt-injection patterns (`ignore all instructions`, etc.)
- Wraps every tool call in `safe_tool_runner()` so exceptions are caught gracefully

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env → set GOOGLE_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/app/apikey

### 3. Run the app
```bash
streamlit run apps/multi-tool-assistant/app.py
```

### 4. Run the tests
```bash
python -m pytest tests/ -v
```

---

## 💡 Example Queries

| Query | Tool used |
|-------|-----------|
| `248 * 17 - 33` | 🔢 Calculator |
| `What is the difference between RAM and ROM?` | 💬 Q&A |
| `Explain more` *(after a Q&A answer)* | 💬 Q&A (memory-aware) |
| *(paste a 50+ word paragraph)* | 📝 Summarizer |

---

## ➕ Extending the Platform

### Add a new tool
1. Create `core/tools/my_tool.py` with a `run(state: dict) -> dict` function
2. Add a node in `core/agents/multi_tool_graph.py`
3. Add a routing label to `ROUTER_PROMPT` in `core/prompts/prompts.py`

### Add a new app
1. Create `apps/my-new-app/app.py`
2. Import from `core.*` — all platform capabilities are immediately available

### Swap the LLM
Edit one line in `core/models/llm.py` — every module picks up the change.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Orchestration | LangGraph (StateGraph) |
| LLM chains | LangChain |
| LLM | Google Gemini 2.5 Flash lite|
| Config | python-dotenv |
| Testing | pytest |

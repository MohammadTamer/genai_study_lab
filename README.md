# GenAI StudyLab — Mini Project

This teaching-focused mini project puts the concepts from the course together:

- LLM
- Prompting
- Prompt Templates
- Zero-shot prompting
- Few-shot prompting
- In-context learning
- LCEL and the `|` pipe
- RunnableLambda
- RunnableParallel
- RunnableSequence concepts
- RAG
- Retriever
- Embeddings
- Vector store
- Hallucination mitigation via grounded prompts
- Agents + tools
- Multi-agent orchestration
- Self-consistency
- FastAPI web application

## Architecture

```text
User
 |
 +--> Basic prompt --> LLM
 |
 +--> RAG --> Retriever --> Vector Store --> LLM
 |
 +--> Agent --> Calculator / Knowledge Search --> LLM
 |
 +--> Multi-Agent Manager --> Research Agent + Math Agent
 |
 +--> Self-consistency --> multiple samples --> majority vote
```

## Requirements

- Python 3.11+
- Ollama installed and running
- A model that fits your machine

### Install

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
copy .env.example .env
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

### Pull local models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

You can change the model names in `.env`.

## Run CLI

```bash
python -m src.main
```

## Run FastAPI

```bash
python app.py
```

or with Uvicorn:

```bash
uvicorn app:app --reload
```

Then open Swagger UI:

http://127.0.0.1:8000/docs (or http://127.0.0.1:8000 which redirects to docs)

## Run Gradio Web UI

```bash
python gradio_app.py
```

Then open the interactive interface in your browser:

http://127.0.0.1:7860

## What each file teaches

- `src/prompts.py`: prompt templates, few-shot, grounded prompts.
- `src/chains.py`: LCEL, pipe operator, RunnableLambda, RunnableParallel.
- `src/rag.py`: chunking, embeddings, vector store, retriever.
- `src/tools.py`: tool definitions and safe calculator.
- `src/agents.py`: current LangChain agent API plus a simple multi-agent pattern.
- `src/self_consistency.py`: multiple independent samples + majority vote.
- `app.py`: FastAPI integration and endpoints.
- `gradio_app.py`: Gradio interactive web UI for testing all modes.

## Production notes

The project uses `InMemoryVectorStore` so it is easy to run. For production, replace it with a persistent vector store such as FAISS, Pinecone, PGVector, Qdrant, or another supported backend. Add auth, retries, timeouts, evaluation, tracing, guardrails, and persistent state.

Fine-tuning is explained conceptually but not performed; actually fine-tuning a model would require a domain dataset and a training workflow.

Chain-of-thought is discussed conceptually; this project asks for concise explanations rather than exposing hidden internal reasoning.

## Exercises

1. Add a third tool.
2. Replace the in-memory vector store with FAISS.
3. Add source citations to the UI.
4. Add structured JSON output.
5. Add conversation history.
6. Add a LangGraph workflow for a more explicit multi-agent graph.
7. Create a benchmark set to compare basic prompting vs RAG.
"# genai_study_lab" 

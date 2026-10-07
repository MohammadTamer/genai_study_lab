from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

from src.agents import build_single_agent
from src.chains import basic_chain, build_rag_chain
from src.rag import get_retriever
from src.self_consistency import self_consistent_answer

app = FastAPI(
    title="GenAI StudyLab API",
    description="Interactive API for GenAI: Prompting, RAG, Agents, and Self-Consistency",
    version="1.0.0",
)

retriever, _, _ = get_retriever()
rag_chain = build_rag_chain(retriever)
agent = build_single_agent(retriever)


class AskRequest(BaseModel):
    question: str = Field(..., description="The user query or prompt", example="What is RAG?")
    mode: str = Field(
        default="rag",
        description="Execution mode: 'rag', 'basic', 'agent', or 'self_consistency'",
        example="rag",
    )


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the AI model")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.post(
    "/api/ask",
    response_model=AskResponse,
    summary="Ask Question",
    description="Send a question to the AI engine using one of the available modes: basic, rag, agent, or self_consistency.",
)
def ask(payload: AskRequest):
    question = payload.question.strip()
    mode = payload.mode
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        if mode == "basic":
            answer = basic_chain.invoke({"question": question})
        elif mode == "rag":
            answer = rag_chain.invoke(question)
        elif mode == "agent":
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            answer = result["messages"][-1].content
        elif mode == "self_consistency":
            result = self_consistent_answer(question)
            answer = (
                f"Final answer: {result['answer']}\n"
                f"Votes: {result['votes']}/{len(result['candidates'])}\n\n"
                + "\n---\n".join(result["samples"])
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode '{mode}'. Supported modes: basic, rag, agent, self_consistency.",
            )

        return AskResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    from src.config import APP_HOST, APP_PORT

    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)

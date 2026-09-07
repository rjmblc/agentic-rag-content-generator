from fastapi import FastAPI
from pydantic import BaseModel

from src.app.rag.agent import run_agent


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic RAG Content Generator",
    description="API for generating content using an Agentic RAG pipeline.",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    question: str
    top_k: int = 5


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# GENERATE CONTENT
# ============================================================

@app.post("/generate")
def generate(request: GenerateRequest):

    result = run_agent(
        question=request.question,
        top_k=request.top_k
    )

    return {
        "question": result["question"],
        "context_grade": result["context_grade"],
        "answer": result["answer"]
    }
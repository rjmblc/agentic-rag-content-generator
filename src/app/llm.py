import os

from langchain_ollama import ChatOllama


MODEL_NAME = "gemma2:2b"
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)


def create_llm():

    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=500,
        repeat_penalty=1.2,
    )
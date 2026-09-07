from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH="data/vectorstore"

EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings():
    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings

def create_vector_store(chunks):
    embeddings=create_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    Path(VECTORSTORE_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(VECTORSTORE_PATH)

    print(
        f"Vector store saved to:{VECTORSTORE_PATH}"
    )

    return vector_store

def load_vector_store():
    embeddings = create_embeddings()

    vector_store=FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store
from src.app.rag.vector_store import load_vector_store


def get_vector_store():
    return load_vector_store()


def search_documents(query: str, top_k: int = 5):

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=top_k
    )

    return results
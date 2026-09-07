from src.app.rag.loader import load_documents, split_documents
from src.app.rag.vector_store import create_vector_store


DOCUMENT_PATH = "src/data/documents"


def main():

    print("===================================")
    print("RAG INGESTION")
    print("===================================")

    documents = load_documents(DOCUMENT_PATH)

    print(f"Documents loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    create_vector_store(chunks)

    print("===================================")
    print("INGESTION COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
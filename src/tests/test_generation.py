from src.app.rag.retriever import search_documents
from src.app.generation import generate_content


def main():

    question = """
    Explain the key dimensions of clinical data quality
    and why they are important.
    """

    print("\nRetrieving documents...")

    results = search_documents(
        query=question,
        top_k=5
    )

    documents = [
        document
        for document, score in results
    ]

    print("Documents retrieved.")

    print("\nGenerating content...")

    response = generate_content(
        question=question,
        documents=documents
    )

    print("\n==============================")
    print("GENERATED CONTENT")
    print("==============================")

    print(response)


if __name__ == "__main__":
    main()
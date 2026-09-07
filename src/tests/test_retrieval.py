from src.app.rag.retriever import search_documents


def main():

    query = """
    What are the important aspects of data quality
    in clinical trials?
    """

    results = search_documents(
        query=query,
        top_k=5
    )

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================")

    for i, (document, score) in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print(f"Score: {score}")

        print("\nContent:")
        print(document.page_content[:1000])

        print("\nMetadata:")
        print(document.metadata)


if __name__ == "__main__":
    main()
from src.app.rag.retriever import search_documents


EVALUATION_DATASET = [
    {
        "question": "What are the key dimensions of clinical data quality?",
        "relevant_keywords": [
            "completeness",
            "accuracy",
            "structure",
            "standardization",
            "security"
        ]
    },
    {
        "question": "Why is completeness important in clinical data quality?",
        "relevant_keywords": [
            "completeness",
            "missing",
            "data quality"
        ]
    },
    {
        "question": "Why is accuracy important in clinical data?",
        "relevant_keywords": [
            "accuracy",
            "reliable",
            "data quality"
        ]
    }
]


def evaluate_retrieval(question, relevant_keywords, top_k=5):

    results = search_documents(
        query=question,
        top_k=top_k
    )

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    relevant_count = 0

    for rank, (document, score) in enumerate(results, start=1):

        content = document.page_content.lower()

        is_relevant = any(
            keyword.lower() in content
            for keyword in relevant_keywords
        )

        if is_relevant:
            relevant_count += 1

        print(f"\nRank {rank}")
        print(f"Score: {score}")
        print(f"Relevant: {is_relevant}")
        print(f"Content: {document.page_content[:300]}...")

    precision_at_k = relevant_count / top_k

    print("\nRetrieval Metrics")
    print("-----------------")
    print(f"Precision@{top_k}: {precision_at_k:.2f}")

    return precision_at_k


def main():

    print("\n==============================")
    print("RAG RETRIEVAL EVALUATION")
    print("==============================")

    scores = []

    for item in EVALUATION_DATASET:

        score = evaluate_retrieval(
            question=item["question"],
            relevant_keywords=item["relevant_keywords"],
            top_k=5
        )

        scores.append(score)

    average_precision = sum(scores) / len(scores)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Average Precision@5: {average_precision:.2f}")


if __name__ == "__main__":
    main()
from src.app.rag.retriever import search_documents
from src.app.generation import generate_content
from src.app.evaluation.generation_evaluator import (
    evaluate_faithfulness,
    evaluate_relevance,
    evaluate_quality
)


def main():

    question = """
    Explain the key dimensions of clinical data quality
    and why they are important.
    """

    print("\n============================================================")
    print("RAG GENERATION EVALUATION")
    print("============================================================")

    # ============================================================
    # 1. RETRIEVAL
    # ============================================================

    print("\nRetrieving documents...")

    results = search_documents(
        query=question,
        top_k=5
    )

    documents = [
        document
        for document, score in results
    ]

    print(f"Retrieved {len(documents)} documents.")

    # ============================================================
    # 2. BUILD CONTEXT
    # ============================================================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # ============================================================
    # 3. GENERATION
    # ============================================================

    print("\nGenerating answer...")

    answer = generate_content(
        question=question,
        documents=documents
    )

    print("\n============================================================")
    print("GENERATED ANSWER")
    print("============================================================")

    print(answer)

    # ============================================================
    # 4. FAITHFULNESS EVALUATION
    # ============================================================

    print("\nEvaluating faithfulness...")

    faithfulness_result = evaluate_faithfulness(
        context=context,
        answer=answer
    )

    print("\n============================================================")
    print("FAITHFULNESS EVALUATION")
    print("============================================================")

    print(
        f"Score: {faithfulness_result['faithfulness_score']}"
    )

    print(
        f"Reason: {faithfulness_result['reason']}"
    )

    # ============================================================
    # 5. RELEVANCE EVALUATION
    # ============================================================

    print("\nEvaluating relevance...")

    relevance_result = evaluate_relevance(
        question=question,
        answer=answer
    )

    print("\n============================================================")
    print("RELEVANCE EVALUATION")
    print("============================================================")

    print(
        f"Score: {relevance_result['relevance_score']}"
    )

    print(
        f"Reason: {relevance_result['reason']}"
    )

    # ============================================================
    # 6. GENERATION QUALITY EVALUATION
    # ============================================================

    print("\nEvaluating generation quality...")

    quality_result = evaluate_quality(
        answer=answer
    )

    print("\n============================================================")
    print("GENERATION QUALITY EVALUATION")
    print("============================================================")

    print(
        f"Score: {quality_result['quality_score']}"
    )

    print(
        f"Reason: {quality_result['reason']}"
    )

    # ============================================================
    # 7. FINAL SUMMARY
    # ============================================================

    print("\n============================================================")
    print("FINAL GENERATION EVALUATION")
    print("============================================================")

    print(
        f"Faithfulness : "
        f"{faithfulness_result['faithfulness_score']}"
    )

    print(
        f"Relevance    : "
        f"{relevance_result['relevance_score']}"
    )

    print(
        f"Quality      : "
        f"{quality_result['quality_score']}"
    )

    print("============================================================")


if __name__ == "__main__":
    main()
from src.app.rag.agent import run_agent


def main():

    question = """
    Explain the key dimensions of clinical data quality
    and why they are important.
    """

    print("\n============================================================")
    print("AGENTIC RAG TEST")
    print("============================================================")

    print("\nQuestion:")
    print(question)

    # ========================================================
    # RUN AGENT
    # ========================================================

    print("\nRunning agent...")

    result = run_agent(
        question=question,
        top_k=5
    )

    # ========================================================
    # CONTEXT GRADE
    # ========================================================

    print("\n============================================================")
    print("CONTEXT GRADING")
    print("============================================================")

    print(
        f"Decision: {result['context_grade']}"
    )

    # ========================================================
    # GENERATED ANSWER
    # ========================================================

    print("\n============================================================")
    print("AGENT ANSWER")
    print("============================================================")

    print(result["answer"])

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n============================================================")
    print("AGENTIC RAG RESULT")
    print("============================================================")

    if result["context_grade"] == "RELEVANT":
        print("Context was relevant.")
        print("Answer was generated.")
    else:
        print("Context was not relevant.")
        print("Answer generation was skipped.")

    print("============================================================")


if __name__ == "__main__":
    main()
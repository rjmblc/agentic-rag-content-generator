from src.app.llm import create_llm
from src.app.rag.retriever import search_documents
from src.app.generation import generate_content


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 2


# ============================================================
# CONTEXT GRADING PROMPT
# ============================================================

CONTEXT_GRADING_PROMPT = """
You are evaluating whether retrieved documents are useful for answering
a user's question in a Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether the retrieved context contains
information that is relevant to the user's question.

IMPORTANT RULES:

1. Read the USER QUESTION.
2. Read the RETRIEVED CONTEXT.
3. Determine whether the context contains useful information
   for answering the question.
4. Do not use outside knowledge.
5. Return RELEVANT if the context contains meaningful information
   that can help answer the question.
6. Return NOT_RELEVANT if the context does not meaningfully help
   answer the question.

Return EXACTLY one of:

RELEVANT
NOT_RELEVANT

================ USER QUESTION ================

{question}

================ RETRIEVED CONTEXT ================

{context}

===============================================
"""


# ============================================================
# QUERY REWRITING PROMPT
# ============================================================

QUERY_REWRITE_PROMPT = """
You are a query rewriting component in an Agentic RAG system.

The original user question did not retrieve sufficiently relevant
documents.

Rewrite the question into a better search query that is more likely
to retrieve useful documents from the knowledge base.

IMPORTANT RULES:

1. Preserve the original user's intent.
2. Identify the important concepts and keywords.
3. Make the query specific and information-rich.
4. Do not answer the question.
5. Do not add facts that are not present in the original question.
6. Return ONLY the rewritten search query.
7. Do not use quotation marks.
8. Do not add explanations.

================ ORIGINAL QUESTION ================

{question}

====================================================
"""


# ============================================================
# GRADE RETRIEVED CONTEXT
# ============================================================

def grade_context(
    question: str,
    documents
) -> str:

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = CONTEXT_GRADING_PROMPT.format(
        question=question,
        context=context
    )

    llm = create_llm()

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    if "NOT_RELEVANT" in result:
        return "NOT_RELEVANT"

    if "RELEVANT" in result:
        return "RELEVANT"

    return "NOT_RELEVANT"


# ============================================================
# REWRITE QUERY
# ============================================================

def rewrite_query(
    question: str
) -> str:

    prompt = QUERY_REWRITE_PROMPT.format(
        question=question
    )

    llm = create_llm()

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    return rewritten_query


# ============================================================
# AGENTIC RAG
# ============================================================

def run_agent(
    question: str,
    top_k: int = 5
) -> dict:

    current_question = question

    attempts = []

    # ========================================================
    # RETRIEVAL + GRADING LOOP
    # ========================================================

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"\nAgent attempt {attempt}/{MAX_RETRIES}"
        )

        # ----------------------------------------------------
        # 1. RETRIEVE
        # ----------------------------------------------------

        print("Retrieving documents...")

        documents_with_scores = search_documents(
            query=current_question,
            top_k=top_k
        )

        documents = [
            document
            for document, score in documents_with_scores
        ]

        print(
            f"Retrieved {len(documents)} documents."
        )

        # ----------------------------------------------------
        # 2. GRADE CONTEXT
        # ----------------------------------------------------

        print("Grading retrieved context...")

        context_grade = grade_context(
            question=current_question,
            documents=documents
        )

        print(
            f"Context grade: {context_grade}"
        )

        # ----------------------------------------------------
        # 3. RECORD ATTEMPT
        # ----------------------------------------------------

        attempts.append({
            "attempt": attempt,
            "query": current_question,
            "context_grade": context_grade,
            "documents": documents
        })

        # ----------------------------------------------------
        # 4. RELEVANT CONTEXT
        # ----------------------------------------------------

        if context_grade == "RELEVANT":

            print(
                "Agent decision: Context is relevant."
            )

            print("Generating answer...")

            answer = generate_content(
                question=question,
                documents=documents
            )

            return {
                "question": question,
                "documents": documents,
                "context_grade": context_grade,
                "answer": answer,
                "attempts": attempts
            }

        # ----------------------------------------------------
        # 5. NOT RELEVANT
        # ----------------------------------------------------

        print(
            "Agent decision: Context is not relevant."
        )

        # ----------------------------------------------------
        # 6. CHECK RETRY LIMIT
        # ----------------------------------------------------

        if attempt >= MAX_RETRIES:
            break

        # ----------------------------------------------------
        # 7. REWRITE QUERY
        # ----------------------------------------------------

        print("Rewriting query...")

        current_question = rewrite_query(
            question=current_question
        )

        print(
            f"Rewritten query: {current_question}"
        )

    # ========================================================
    # MAX RETRIES EXCEEDED
    # ========================================================

    print(
        "\nAgent could not find relevant context "
        "within the maximum number of attempts."
    )

    last_attempt = attempts[-1]

    return {
        "question": question,
        "documents": last_attempt["documents"],
        "context_grade": "NOT_RELEVANT",
        "answer": (
            "I could not find sufficient relevant information "
            "in the retrieved documents to answer the question."
        ),
        "attempts": attempts
    }
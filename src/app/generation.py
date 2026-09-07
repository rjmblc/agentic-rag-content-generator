from src.app.llm import create_llm
from src.app.prompts import CONTENT_GENERATION_PROMPT


def generate_content(
    question: str,
    documents
):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = CONTENT_GENERATION_PROMPT.format_messages(
        question=question,
        context=context
    )

    llm = create_llm()

    response = llm.invoke(prompt)

    return response.content
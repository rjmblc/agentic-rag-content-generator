from src.app.llm import create_llm


def main():

    print("Creating LLM...")

    llm = create_llm()

    print("Calling Ollama...")

    response = llm.invoke(
        "Say hello in one sentence."
    )

    print("\n==============================")
    print("OLLAMA RESPONSE")
    print("==============================")

    print(response.content)


if __name__ == "__main__":
    main()
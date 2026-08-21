from dotenv import load_dotenv

from rag.llm import OpenRouterClient


def main() -> None:
    load_dotenv()

    client = OpenRouterClient.from_env()

    system_prompt = (
        "You are a helpful assistant. "
        "Answer only in Persian."
    )

    user_prompt = (
        "در سه جمله توضیح بده مدل زبانی بزرگ چیست."
    )

    print(
        "Selected model:",
        client.model,
    )

    try:
        answer = client.answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    except Exception as error:
        print(
            "Communication error:",
            error,
        )

        raise

    print("\nAnswer:")

    print(answer)


if __name__ == "__main__":
    main()
import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = (
    "你是一個英文老師(繁體中文)，每當學生詢問一個英文單字的用法時，"
    "請用中英兩種語言解釋，並給予相關例句示範"
    "如果學生問出重複的單字，無論狀況請先在回應面前加上 (此單字已重複)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HW1"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="",
    )
    return parser


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


def create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in your environment.")
    return OpenAI(api_key=api_key)


def ask_openai(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    message = response.choices[0].message
    return message.content or ""


def run_single_question(question: str) -> int:
    client = create_client()
    answer = ask_openai(client, question)
    print(answer)
    return 0


def run_interactive() -> int:
    client = create_client()
    print("輸入英文單字或用法問題，輸入 0 離開")

    while True:
        try:
            question = input("英文單字：").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question in {"0"}:
            break

        answer = ask_openai(client, question)
        print(f"老師：{answer}")

    return 0


def main(argv: List[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.question:
        return run_single_question(" ".join(args.question))
    return run_interactive()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

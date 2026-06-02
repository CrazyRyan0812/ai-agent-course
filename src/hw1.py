import argparse
import os
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = (
    "你是一個英文老師(繁體中文)，每當學生詢問一個英文單字的用法時，"
    "請用中英兩種語言解釋，並給予相關例句示範"
)

Message = Dict[str, str]


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
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please configure it in your environment."
        )
    return OpenAI(api_key=api_key)


def _message_content(message: object) -> str:
    content = getattr(message, "content", "")
    return content or ""


class ChatSession:
    def __init__(
        self,
        client: OpenAI,
        model_name: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.model_name = model_name or get_model_name()
        self.messages: List[Message] = [{"role": "developer", "content": system_prompt}]

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        answer = _message_content(response.choices[0].message)
        self.messages.append({"role": "assistant", "content": answer})
        return answer


def run_single_question(question: str) -> int:
    client = create_client()
    session = ChatSession(client)
    answer = session.ask(question)
    print(answer)
    return 0


def run_interactive() -> int:
    client = create_client()
    session = ChatSession(client)
    print("輸入英文單字或用法問題，輸入 0 離開。")

    while True:
        try:
            question = input("學生輸入：").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question in {"0"}:
            break

        answer = session.ask(question)
        print(f"老師：{answer}")
        print("================================")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
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
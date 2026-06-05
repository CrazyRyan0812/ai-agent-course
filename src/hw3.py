import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - dependency fallback for tests
    QdrantClient = Any  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = "coffee_hw3"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_ANSWER_MODEL = "gpt-5-mini"

ANSWER_SYSTEM_PROMPT = (
    "你是一位咖啡飲品介紹助理。請根據檢索到的咖啡資料，"
    "用繁體中文回答使用者的問題，先給出推薦，再簡短說明原因，"
    "最後補一個實用的小建議。不得捏造資料庫中沒有的內容。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW3 coffee RAG assistant")
    parser.add_argument("question", nargs="*", help="Optional question to ask directly")
    return parser


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def load_knowledge_base(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("coffee knowledge base must be a list")
    return data


def item_to_text(item: Dict[str, Any]) -> str:
    keywords = item.get("keywords", [])
    if isinstance(keywords, list):
        keyword_text = ", ".join(str(keyword) for keyword in keywords)
    else:
        keyword_text = str(keywords)
    return "\n".join(
        [
            f"名稱：{item.get('name', '')}",
            f"介紹：{item.get('description', '')}",
            f"風味：{item.get('flavor', '')}",
            f"組成：{item.get('ingredients', '')}",
            f"適合對象：{item.get('best_for', '')}",
            f"關鍵字：{keyword_text}",
        ]
    )


def build_context_block(items: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"[{index}] {item.get('name', '')}")
        lines.append(f"介紹：{item.get('description', '')}")
        lines.append(f"風味：{item.get('flavor', '')}")
        lines.append(f"組成：{item.get('ingredients', '')}")
        lines.append(f"適合對象：{item.get('best_for', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def format_search_summary(hits: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, hit in enumerate(hits, start=1):
        payload = hit["payload"]
        lines.append(f"{index}. {payload['name']} (score={hit['score']:.3f})")
        lines.append(f"   {payload['description']}")
    return "\n".join(lines)


def extract_hit_payload(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        return hit["payload"]
    return getattr(hit, "payload")


def extract_hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit["score"])
    return float(getattr(hit, "score", 0.0))


def normalize_query_points_response(response: Any) -> Sequence[Any]:
    if hasattr(response, "points"):
        return response.points
    if hasattr(response, "result"):
        return response.result
    return response


def create_openai_client() -> OpenAI:
    return OpenAI(api_key=get_env("OPENAI_API_KEY"))


def create_qdrant_client() -> QdrantClient:
    url = get_env("QDRANT_URL")
    api_key = get_env("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


@dataclass
class CoffeeRAGService:
    openai_client: Any
    qdrant_client: Any
    collection_name: str = DEFAULT_COLLECTION
    embed_model: str = DEFAULT_EMBED_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL

    def embed(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(model=self.embed_model, input=text)
        return response.data[0].embedding

    def seed(self, knowledge_base: Sequence[Dict[str, Any]]) -> None:
        points = []
        for idx, item in enumerate(knowledge_base, start=1):
            vector = self.embed(item_to_text(item))
            if qmodels is not None:
                points.append(qmodels.PointStruct(id=idx, vector=vector, payload=item))
            else:
                points.append({"id": idx, "vector": vector, "payload": item})
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embed(query)
        query_points = getattr(self.qdrant_client, "query_points", None)
        search = getattr(self.qdrant_client, "search", None)

        if callable(query_points):
            response = query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            hits = normalize_query_points_response(response)
        elif callable(search):
            hits = search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            raise AttributeError(
                "Qdrant client does not provide query_points() or search()."
            )
        return [
            {
                "score": extract_hit_score(hit),
                "payload": extract_hit_payload(hit),
            }
            for hit in hits
        ]

    def answer(self, query: str, limit: int = 3) -> str:
        hits = self.search(query, limit=limit)
        context = build_context_block([hit["payload"] for hit in hits])
        response = self.openai_client.chat.completions.create(
            model=self.answer_model,
            messages=[
                {"role": "developer", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "system", "content": f"檢索到的咖啡資料：\n{context}"},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content or ""


def print_search_results(query: str, limit: int = 3) -> int:
    openai_client = create_openai_client()
    qdrant_client = create_qdrant_client()
    service = CoffeeRAGService(openai_client, qdrant_client)
    hits = service.search(query, limit=limit)
    print(format_search_summary(hits))
    return 0


def run_interactive() -> int:
    openai_client = create_openai_client()
    qdrant_client = create_qdrant_client()
    service = CoffeeRAGService(openai_client, qdrant_client)
    print("輸入咖啡相關問題，輸入 exit 離開。")

    while True:
        try:
            question = input("你：").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = service.answer(question, limit=3)
        print(f"助理：{answer}")

    return 0


def run_single_question(question: str) -> int:
    openai_client = create_openai_client()
    qdrant_client = create_qdrant_client()
    service = CoffeeRAGService(openai_client, qdrant_client)
    answer = service.answer(question, limit=3)
    print(answer)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.question:
        return run_single_question(" ".join(args.question))
    return run_interactive()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

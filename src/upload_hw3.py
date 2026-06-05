import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    QdrantClient = Any  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


DATA_PATH = Path("coffee_hw3.json")
DEFAULT_COLLECTION = "coffee_hw3"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def load_knowledge_base(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("knowledge base must be a list")
    return data


def build_point_text(item: Dict[str, Any]) -> str:
    keywords = item.get("keywords", [])
    keyword_text = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
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


def create_openai_client() -> OpenAI:
    return OpenAI(api_key=get_env("OPENAI_API_KEY"))


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_env("QDRANT_URL"), api_key=get_env("QDRANT_API_KEY"))


@dataclass
class CoffeeUploader:
    openai_client: Any
    qdrant_client: Any
    collection_name: str = DEFAULT_COLLECTION
    embed_model: str = DEFAULT_EMBED_MODEL

    def embed(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(model=self.embed_model, input=text)
        return response.data[0].embedding

    def ensure_collection(self, vector_size: int) -> None:
        if self.qdrant_client.collection_exists(self.collection_name):
            return
        if qmodels is not None:
            vectors_config = qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            )
        else:
            vectors_config = {"size": vector_size, "distance": "Cosine"}
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )

    def seed(self, knowledge_base: Sequence[Dict[str, Any]]) -> None:
        points = []
        for idx, item in enumerate(knowledge_base, start=1):
            vector = self.embed(build_point_text(item))
            if qmodels is not None:
                points.append(qmodels.PointStruct(id=idx, vector=vector, payload=item))
            else:
                points.append({"id": idx, "vector": vector, "payload": item})
        if points:
            self.ensure_collection(vector_size=len(points[0].vector if qmodels is not None else points[0]["vector"]))
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points, wait=True)


def main() -> int:
    load_dotenv()
    knowledge_base = load_knowledge_base(DATA_PATH)
    uploader = CoffeeUploader(
        openai_client=create_openai_client(),
        qdrant_client=create_qdrant_client(),
        collection_name=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )
    uploader.seed(knowledge_base)
    print(f"Uploaded {len(knowledge_base)} coffee items to Qdrant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from hw3 import (
    CoffeeRAGService,
    create_openai_client,
    create_qdrant_client,
    load_knowledge_base,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coffee_hw3.json"


def main() -> int:
    load_dotenv()
    knowledge_base = load_knowledge_base(DATA_PATH)
    service = CoffeeRAGService(
        openai_client=create_openai_client(),
        qdrant_client=create_qdrant_client(),
        collection_name=os.getenv("QDRANT_COLLECTION", "coffee_hw3"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    )
    service.seed(knowledge_base)
    print(f"Seeded {len(knowledge_base)} coffee items into Qdrant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:  # type: ignore[override]
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class SentenceGroup:
    group_name: str
    sentences: Sequence[str]
    analysis_hint: str


EXPERIMENT_GROUPS: Sequence[SentenceGroup] = (
    SentenceGroup(
        group_name="第 1 組：意思相近",
        sentences=("我喜歡貓", "貓咪很可愛", "我養了一隻貓"),
        analysis_hint="這組句子表達的需求接近，預期相似度最高。",
    ),
    SentenceGroup(
        group_name="第 2 組：意思不同",
        sentences=("今天天氣很好", "我要去買菜", "電腦壞了"),
        analysis_hint="這組句子主題差異明顯，預期相似度最低。",
    ),
    SentenceGroup(
        group_name="第 3 組：邊界案例",
        sentences=("我喜歡貓", "今天天氣很好", "天氣變冷時，我喜歡嚕貓"),
        analysis_hint="這組句子有情境關聯，但不是完全同義，適合觀察邊界分數。",
    ),
)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    return dot_product / denominator


def create_openai_client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - dependency fallback
        raise RuntimeError("openai package is required to run HW5.") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=api_key)


def embed_sentences(client: Any, sentences: Sequence[str], model: str) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=list(sentences))
    return [list(item.embedding) for item in response.data]


def build_similarity_matrix(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for vector_a in vectors:
        row = [cosine_similarity(vector_a, vector_b) for vector_b in vectors]
        matrix.append(row)
    return matrix


def build_pairwise_rows(sentences: Sequence[str], matrix: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(sentences)), 2):
        rows.append(
            {
                "pair": f"{left_index + 1} vs {right_index + 1}",
                "left_sentence": sentences[left_index],
                "right_sentence": sentences[right_index],
                "similarity": matrix[left_index][right_index],
            }
        )
    return rows


def _summarize_group(group_name: str, average_similarity: float, analysis_hint: str) -> str:
    if average_similarity >= 0.85:
        verdict = "整體相似度偏高，模型把這組句子視為語意接近。"
    elif average_similarity <= 0.35:
        verdict = "整體相似度偏低，模型把這組句子視為語意差異明顯。"
    else:
        verdict = "整體相似度介於兩端，屬於有關聯但不完全相同的情況。"
    return f"{group_name}：{verdict} {analysis_hint}"


def run_experiment(*, client: Any, model: str = DEFAULT_EMBED_MODEL) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for group in EXPERIMENT_GROUPS:
        vectors = embed_sentences(client, group.sentences, model=model)
        matrix = build_similarity_matrix(vectors)
        pairwise = build_pairwise_rows(group.sentences, matrix)
        pairwise_values = [row["similarity"] for row in pairwise]
        average_similarity = sum(pairwise_values) / len(pairwise_values) if pairwise_values else 1.0
        results.append(
            {
                "group_name": group.group_name,
                "model": model,
                "sentences": list(group.sentences),
                "vectors": vectors,
                "matrix": matrix,
                "pairwise": pairwise,
                "average_similarity": average_similarity,
                "analysis": _summarize_group(
                    group.group_name,
                    average_similarity,
                    group.analysis_hint,
                ),
            }
        )
    return results


def _format_matrix(sentences: Sequence[str], matrix: Sequence[Sequence[float]]) -> List[str]:
    labels = [f"S{i}" for i in range(1, len(sentences) + 1)]
    lines = []
    header = ["    "] + [f"{label:>8}" for label in labels]
    lines.append("".join(header))
    for label, row in zip(labels, matrix):
        lines.append(label.ljust(4) + "".join(f"{value:8.3f}" for value in row))
    return lines


def format_experiment_report(results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("HW5 向量相似度實驗")
    model_name = results[0].get("model", DEFAULT_EMBED_MODEL) if results else DEFAULT_EMBED_MODEL
    lines.append(f"模型：{model_name}")
    lines.append("")

    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['group_name']}")
        lines.append("句子：")
        for sentence_index, sentence in enumerate(result["sentences"], start=1):
            lines.append(f"  {sentence_index}. {sentence}")
        lines.append("相似度矩陣：")
        lines.extend(_format_matrix(result["sentences"], result["matrix"]))
        lines.append("成對相似度：")
        for row in result["pairwise"]:
            lines.append(f"  句子 {row['pair']} = {row['similarity']:.3f}")
        lines.append(f"平均相似度：{result['average_similarity']:.3f}")
        lines.append(f"分析：{result['analysis']}")
        lines.append("")

    if results:
        ranked = sorted(
            results,
            key=lambda item: item["average_similarity"],
            reverse=True,
        )
        lines.append("總結：")
        lines.append(f"  相似度最高：{ranked[0]['group_name']} ({ranked[0]['average_similarity']:.3f})")
        lines.append(f"  相似度最低：{ranked[-1]['group_name']} ({ranked[-1]['average_similarity']:.3f})")
        lines.append("  第 3 組用來觀察模型如何處理語意邊界。")

    return "\n".join(lines)


def main(output_func=print, client: Optional[Any] = None, model: Optional[str] = None) -> int:
    load_dotenv()
    if client is None:
        client = create_openai_client()
    if model is None:
        model = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL

    results = run_experiment(client=client, model=model)
    output_func(format_experiment_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

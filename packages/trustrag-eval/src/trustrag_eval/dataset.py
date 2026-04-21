"""Load and parse synthetic query datasets for TrustRAG benchmarking."""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Query:
    id: str
    text: str
    category: str  # semantic | keyword | hybrid
    ground_truth_chunk_ids: list[str] = field(default_factory=list)
    expected_answer_substring: str | None = None


def load_synthetic_queries(path: str | Path = "eval/synthetic_queries.json") -> list[Query]:
    """Load synthetic queries from JSON file.

    Args:
        path: Path to the synthetic_queries.json file.

    Returns:
        List of Query objects parsed from the dataset.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Query(
            id=q["id"],
            text=q["text"],
            category=q["category"],
            ground_truth_chunk_ids=q.get("ground_truth_chunk_ids", []),
            expected_answer_substring=q.get("expected_answer_substring"),
        )
        for q in data["queries"]
    ]

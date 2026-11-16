"""Search a local directory of text files, showing citation offsets and component scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Document, HybridIndex, evaluate


def load_documents(directory: Path) -> list[Document]:
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    return [
        Document(path.relative_to(directory).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("*.txt"))
        if path.is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainable local text retrieval")
    parser.add_argument("--directory", type=Path, default=Path("examples/corpus"))
    parser.add_argument("--query", default="How does an electric motor drive a pump?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--evaluation", type=Path, help="Optional JSON list of [query, source] pairs")
    args = parser.parse_args()
    index = HybridIndex(load_documents(args.directory))
    print(f"Indexed {len(index.documents)} files into {len(index.passages)} passages")
    for rank, hit in enumerate(index.search(args.query, args.top_k), 1):
        p = hit.passage
        print(f"\n#{rank} {p.id} | BM25={hit.bm25:.3f} char={hit.char_similarity:.3f}"
              f" RRF={hit.fused:.5f}\n{p.text[:350]}")
    if args.evaluation:
        questions = json.loads(args.evaluation.read_text(encoding="utf-8"))
        print("\nEvaluation:", json.dumps(evaluate(index, questions, args.top_k), indent=2))


if __name__ == "__main__":
    main()

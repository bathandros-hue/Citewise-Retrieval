"""Transparent BM25 and character-gram retrieval with source offsets."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from collections.abc import Iterable

WORD = re.compile(r"\b[\w]+\b", re.UNICODE)
NONSPACE = re.compile(r"\S+", re.UNICODE)


def terms(text: str) -> list[str]:
    return [x.group().lower() for x in WORD.finditer(text)]


def grams(text: str) -> Counter[str]:
    normalized = " ".join(terms(text))
    return Counter(normalized[i : i + 3] for i in range(max(0, len(normalized) - 2)))


@dataclass(frozen=True)
class Document:
    name: str
    text: str


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    bm25: float
    char_similarity: float
    fused: float


def chunk_document(document: Document, size: int = 80, overlap: int = 16) -> list[Passage]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    matches = list(NONSPACE.finditer(document.text))
    chunks: list[Passage] = []
    for i in range(0, len(matches), size - overlap):
        selected = matches[i : i + size]
        if not selected:
            break
        start, end = selected[0].start(), selected[-1].end()
        chunks.append(Passage(
            f"{document.name}:{start}-{end}", document.name, start, end,
            document.text[start:end],
        ))
        if i + size >= len(matches):
            break
    return chunks


class HybridIndex:
    def __init__(self, documents: Iterable[Document], chunk_size: int = 80, overlap: int = 16):
        self.documents = list(documents)
        names = [doc.name for doc in self.documents]
        if len(names) != len(set(names)):
            raise ValueError("Document names must be unique")
        self.passages = [
            chunk for doc in self.documents
            for chunk in chunk_document(doc, chunk_size, overlap)
        ]
        self.tokens = [Counter(terms(p.text)) for p in self.passages]
        self.char_vectors = [grams(p.text) for p in self.passages]
        self.df = Counter(t for tokens in self.tokens for t in tokens)
        self.avg_length = (
            sum(sum(c.values()) for c in self.tokens) / len(self.tokens)
            if self.tokens else 0.0
        )

    def _bm25(self, query: list[str], i: int) -> float:
        if not self.passages:
            return 0.0
        count = self.tokens[i]
        length = sum(count.values())
        score = 0.0
        for token in set(query):
            tf = count[token]
            if not tf:
                continue
            idf = math.log(1 + (len(self.passages) - self.df[token] + 0.5)
                           / (self.df[token] + 0.5))
            score += idf * (tf * 2.2) / (
                tf + 1.2 * (0.25 + 0.75 * length / max(self.avg_length, 1e-9))
            )
        return score

    def _char_score(self, query: Counter[str], i: int) -> float:
        candidate = self.char_vectors[i]
        dot = sum(v * candidate[k] for k, v in query.items())
        denom = math.sqrt(sum(v * v for v in query.values())) * math.sqrt(
            sum(v * v for v in candidate.values())
        )
        return dot / denom if denom else 0.0

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be positive")
        if not query.strip() or not self.passages:
            return []
        tokens, query_grams = terms(query), grams(query)
        bm = [self._bm25(tokens, i) for i in range(len(self.passages))]
        char = [self._char_score(query_grams, i) for i in range(len(self.passages))]
        lexical_order = sorted(range(len(bm)), key=lambda i: (-bm[i], i))
        char_order = sorted(range(len(char)), key=lambda i: (-char[i], i))
        lexical_rank = {i: rank for rank, i in enumerate(lexical_order, 1)}
        char_rank = {i: rank for rank, i in enumerate(char_order, 1)}
        hits = [
            SearchHit(p, bm[i], char[i],
                      1 / (60 + lexical_rank[i]) + 1 / (60 + char_rank[i]))
            for i, p in enumerate(self.passages)
            if bm[i] > 0 or char[i] > 0
        ]
        return sorted(
            hits, key=lambda h: (-h.fused, -h.bm25, h.passage.id)
        )[:k]


def evaluate(index: HybridIndex, questions: Iterable[tuple[str, str]], k: int = 3) -> dict:
    """Document-level Recall@k and reciprocal rank; not a generation-quality metric."""
    cases = list(questions)
    if not cases:
        raise ValueError("Need at least one labeled question")
    recall, reciprocal = 0, 0.0
    for question, expected_document in cases:
        sources = [hit.passage.source for hit in index.search(question, k)]
        if expected_document in sources:
            recall += 1
            reciprocal += 1 / (sources.index(expected_document) + 1)
    return {"questions": len(cases), f"recall@{k}": recall / len(cases),
            f"mrr@{k}": reciprocal / len(cases)}

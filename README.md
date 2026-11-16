# Citewise Retrieval

A local, explainable retrieval engine for text documents. Every result includes an exact source character range, BM25 score, character-trigram cosine similarity, and reciprocal-rank fusion score.

**Quickstart** (Python 3.10+, no runtime dependencies):

    python -m unittest discover -s tests -v
    python -m citewise.cli --directory examples/corpus --query "What does an impeller do?"
    python -m citewise.cli --directory examples/corpus --query "solar collector losses" --evaluation examples/questions.json

## How it works

Documents are chunked by whitespace-token windows while preserving character positions. BM25 captures exact term relevance; character trigrams add some tolerance for spelling differences. Reciprocal-rank fusion combines both rankings. Hits contain source offsets so a caller can verify evidence directly.

## Evaluation

Labeled question/source pairs report document-level Recall@k and MRR@k. This tool does **not** generate answers, claim semantic embedding quality, or guarantee source factuality.

## Limitations

Text files only; in-memory index; character trigrams do not replace semantic embeddings, and scores are not probabilities. Designed for inspectable local retrieval and a foundation for RAG applications.

## GitHub topics

rag, information-retrieval, bm25, reciprocal-rank-fusion, explainable-ai, python

## License

MIT; see LICENSE.

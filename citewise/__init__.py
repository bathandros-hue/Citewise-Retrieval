"""Evidence-first local document retrieval."""
from .engine import Document, HybridIndex, Passage, SearchHit, evaluate

__all__ = ["Document", "HybridIndex", "Passage", "SearchHit", "evaluate"]

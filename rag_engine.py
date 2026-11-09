"""Page-aware text chunking, embeddings, and FAISS retrieval."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_processor import PageText


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 35


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_number: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_number: int
    score: float


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    """Load the sentence embedding model once per application process."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def split_pages_into_chunks(
    pages: Sequence[PageText],
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Split each page independently so every chunk keeps its PDF page source."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between zero and chunk_words - 1.")

    chunks: list[TextChunk] = []
    step = chunk_words - overlap_words
    for page in pages:
        words = page.text.split()
        for start in range(0, len(words), step):
            part = words[start : start + chunk_words]
            if part:
                chunks.append(
                    TextChunk(text=" ".join(part), page_number=page.page_number)
                )
            if start + chunk_words >= len(words):
                break
    return chunks


class RAGEngine:
    """In-memory FAISS index for a single uploaded tender."""

    def __init__(
        self,
        pages: Sequence[PageText],
        embedding_model: SentenceTransformer | None = None,
    ) -> None:
        self.chunks = split_pages_into_chunks(pages)
        if not self.chunks:
            raise ValueError("No readable page text is available to index.")

        self._embedding_model = embedding_model or _load_embedding_model()
        vectors = self._encode([chunk.text for chunk in self.chunks])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the most similar chunks and their cosine-similarity scores."""
        if not question.strip():
            raise ValueError("Enter a question to search the tender.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self._encode([question])
        scores, indices = self._index.search(
            query_vector, min(top_k, len(self.chunks))
        )
        return [
            RetrievedChunk(
                text=self.chunks[int(index)].text,
                page_number=self.chunks[int(index)].page_number,
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

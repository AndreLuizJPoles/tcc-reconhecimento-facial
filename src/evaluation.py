"""Comparação de embeddings faciais por similaridade de cosseno."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.gallery import GalleryRecord

# Threshold provisório, comum em demos do InsightFace/ArcFace.
# Deve ser calibrado depois com uma base de validação; não é um valor científico deste TCC.
DEFAULT_THRESHOLD = 0.5


@dataclass
class ComparisonResult:
    similarity: float
    is_match: bool
    threshold: float


@dataclass
class IdentityCandidate:
    identity: str
    similarity: float
    image_path: str


@dataclass
class IdentificationResult:
    matched: bool
    identity: str | None
    similarity: float
    threshold: float
    best_gallery_path: str | None
    candidates: list[IdentityCandidate]


class FaceComparator:
    """Compara dois embeddings ArcFace pela similaridade de cosseno."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = float(threshold)

    def cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        vector_a = self._as_1d(embedding_a)
        vector_b = self._as_1d(embedding_b)
        if vector_a.size != vector_b.size:
            raise ValueError(
                f"Dimensões incompatíveis: {vector_a.size} e {vector_b.size}."
            )

        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Não é possível comparar embeddings nulos.")

        similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        return float(np.clip(similarity, -1.0, 1.0))

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> ComparisonResult:
        similarity = self.cosine_similarity(embedding_a, embedding_b)
        return ComparisonResult(
            similarity=similarity,
            is_match=similarity >= self.threshold,
            threshold=self.threshold,
        )

    def identify(
        self,
        query_embedding: np.ndarray,
        records: list[GalleryRecord],
        top_k: int = 5,
    ) -> IdentificationResult:
        if not records:
            raise ValueError("A galeria está vazia. Cadastre identidades antes de buscar.")

        best_by_identity: dict[str, IdentityCandidate] = {}

        for record in records:
            similarity = self.cosine_similarity(query_embedding, record.embedding)
            current = best_by_identity.get(record.identity)
            if current is None or similarity > current.similarity:
                best_by_identity[record.identity] = IdentityCandidate(
                    identity=record.identity,
                    similarity=similarity,
                    image_path=record.image_path,
                )

        ranking = sorted(
            best_by_identity.values(),
            key=lambda candidate: candidate.similarity,
            reverse=True,
        )
        best = ranking[0]
        matched = best.similarity >= self.threshold
        return IdentificationResult(
            matched=matched,
            identity=best.identity if matched else None,
            similarity=best.similarity,
            threshold=self.threshold,
            best_gallery_path=best.image_path if matched else None,
            candidates=ranking[:top_k],
        )

    def _as_1d(self, embedding: np.ndarray) -> np.ndarray:
        if embedding is None:
            raise ValueError("O embedding é nulo.")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("O embedding está vazio.")
        return vector

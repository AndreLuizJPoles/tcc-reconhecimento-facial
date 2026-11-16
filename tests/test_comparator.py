"""Testes unitários do FaceComparator (sem modelo InsightFace)."""

import numpy as np
import pytest

from src.evaluation import FaceComparator


def test_identical_embeddings_have_similarity_one() -> None:
    embedding = np.array([0.3, 0.4, 0.0], dtype=np.float32)
    comparator = FaceComparator(threshold=0.5)
    result = comparator.compare(embedding, embedding)
    assert result.similarity == pytest.approx(1.0)
    assert result.is_match is True


def test_orthogonal_embeddings_have_low_similarity() -> None:
    embedding_a = np.array([1.0, 0.0], dtype=np.float32)
    embedding_b = np.array([0.0, 1.0], dtype=np.float32)
    similarity = FaceComparator().cosine_similarity(embedding_a, embedding_b)
    assert similarity == pytest.approx(0.0)


def test_opposite_embeddings_have_negative_similarity() -> None:
    embedding_a = np.array([1.0, 1.0], dtype=np.float32)
    embedding_b = np.array([-1.0, -1.0], dtype=np.float32)
    similarity = FaceComparator().cosine_similarity(embedding_a, embedding_b)
    assert similarity == pytest.approx(-1.0)


def test_incompatible_dimensions_raise_error() -> None:
    embedding_a = np.array([1.0, 0.0], dtype=np.float32)
    embedding_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Dimensões incompatíveis"):
        FaceComparator().cosine_similarity(embedding_a, embedding_b)


def test_threshold_is_configurable() -> None:
    embedding_a = np.array([1.0, 0.0], dtype=np.float32)
    embedding_b = np.array([0.8, 0.6], dtype=np.float32)
    similarity = FaceComparator().cosine_similarity(embedding_a, embedding_b)
    low = FaceComparator(threshold=similarity - 0.05).compare(embedding_a, embedding_b)
    high = FaceComparator(threshold=similarity + 0.05).compare(embedding_a, embedding_b)
    assert low.is_match is True
    assert high.is_match is False


def test_none_embedding_raises_error() -> None:
    with pytest.raises(ValueError, match="nulo"):
        FaceComparator().cosine_similarity(None, np.array([1.0], dtype=np.float32))


def test_empty_embedding_raises_error() -> None:
    with pytest.raises(ValueError, match="vazio"):
        FaceComparator().cosine_similarity(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )


def test_zero_embedding_raises_error() -> None:
    with pytest.raises(ValueError, match="nulos"):
        FaceComparator().cosine_similarity(
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )

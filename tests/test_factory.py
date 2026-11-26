"""Testes do protocolo FaceEmbedder e da factory get_embedder."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embedders.arcface import ArcFaceExtractor
from src.embedders.base import FaceEmbedder
from src.models import get_embedder


def test_get_embedder_returns_arcface_without_adapter() -> None:
    app = MagicMock()
    app.get.return_value = []
    embedder = get_embedder(model_name="arcface", adapter="none", app=app)
    assert isinstance(embedder, ArcFaceExtractor)
    assert embedder.name == "arcface"
    assert isinstance(embedder, FaceEmbedder)
    detections = embedder.extract(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections == []


def test_get_embedder_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        get_embedder(model_name="facenet")


def test_cosface_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="CosFace"):
        get_embedder(model_name="cosface")


def test_auraface_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="AuraFace"):
        get_embedder(model_name="auraface")

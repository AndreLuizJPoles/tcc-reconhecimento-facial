"""Testes do extrator e do carregamento de imagem, com mocks do InsightFace."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models import ArcFaceExtractor
from src.preprocessing import ImageLoadError, load_image, validate_image_array


def test_nonexistent_image_raises_error(tmp_path) -> None:
    missing = tmp_path / "nao_existe.jpg"
    with pytest.raises(ImageLoadError, match="inexistente"):
        load_image(missing)


def test_directory_path_raises_error(tmp_path) -> None:
    with pytest.raises(ImageLoadError, match="não é um arquivo"):
        load_image(tmp_path)


def test_none_image_raises_error() -> None:
    with pytest.raises(ImageLoadError, match="nulo"):
        validate_image_array(None)


def test_empty_array_raises_error() -> None:
    with pytest.raises(ImageLoadError, match="vazia"):
        validate_image_array(np.array([], dtype=np.uint8))


def test_extract_returns_empty_list_when_no_faces() -> None:
    app = MagicMock()
    app.get.return_value = []
    extractor = ArcFaceExtractor(app=app)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = extractor.extract(image)
    assert detections == []
    app.get.assert_called_once()


def test_extract_supports_multiple_faces() -> None:
    app = MagicMock()
    app.get.return_value = [
        SimpleNamespace(
            embedding=np.array([3.0, 4.0], dtype=np.float32),
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
            det_score=0.9,
        ),
        SimpleNamespace(
            normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
            bbox=np.array([11, 0, 20, 10], dtype=np.float32),
            det_score=0.7,
        ),
    ]
    extractor = ArcFaceExtractor(app=app)
    detections = extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].embedding == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))
    assert detections[1].embedding == pytest.approx(np.array([1.0, 0.0], dtype=np.float32))
    assert detections[0].det_score == pytest.approx(0.9)
    assert detections[1].bbox.tolist() == [11.0, 0.0, 20.0, 10.0]


def test_extract_rejects_invalid_image() -> None:
    extractor = ArcFaceExtractor(app=MagicMock())
    with pytest.raises(ImageLoadError):
        extractor.extract(None)

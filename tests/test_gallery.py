"""Testes da galeria e da identificação 1:N (sem InsightFace real)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.evaluation import FaceComparator
from src.gallery import FaceGallery, GalleryRecord
from src.models import ArcFaceExtractor


def _write_jpg(path) -> None:
    blank = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok
    path.write_bytes(encoded.tobytes())


def _record(identity: str, values: list[float], path: str = "x.jpg") -> GalleryRecord:
    return GalleryRecord(
        identity=identity,
        image_path=path,
        embedding=np.array(values, dtype=np.float32),
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        det_score=0.9,
    )


def test_identify_returns_best_identity_above_threshold() -> None:
    records = [
        _record("joao", [1.0, 0.0], "joao.jpg"),
        _record("maria", [0.0, 1.0], "maria.jpg"),
    ]
    result = FaceComparator(threshold=0.4).identify(
        np.array([0.9, 0.1], dtype=np.float32),
        records,
    )
    assert result.matched is True
    assert result.identity == "joao"
    assert result.best_gallery_path == "joao.jpg"
    assert result.candidates[0].identity == "joao"


def test_identify_rejects_below_threshold() -> None:
    records = [_record("joao", [1.0, 0.0])]
    result = FaceComparator(threshold=0.9).identify(
        np.array([0.0, 1.0], dtype=np.float32),
        records,
    )
    assert result.matched is False
    assert result.identity is None
    assert result.best_gallery_path is None
    assert result.similarity == pytest.approx(0.0)


def test_identify_uses_max_score_per_identity() -> None:
    records = [
        _record("joao", [1.0, 0.0], "joao_a.jpg"),
        _record("joao", [0.0, 1.0], "joao_b.jpg"),
    ]
    result = FaceComparator(threshold=0.5).identify(
        np.array([1.0, 0.0], dtype=np.float32),
        records,
    )
    assert result.matched is True
    assert result.identity == "joao"
    assert result.similarity == pytest.approx(1.0)
    assert result.best_gallery_path == "joao_a.jpg"


def test_identify_empty_gallery_raises() -> None:
    with pytest.raises(ValueError, match="vazia"):
        FaceComparator().identify(np.array([1.0, 0.0], dtype=np.float32), [])


def test_enroll_from_identity_folders(tmp_path) -> None:
    joao = tmp_path / "joao"
    maria = tmp_path / "maria"
    joao.mkdir()
    maria.mkdir()
    _write_jpg(joao / "1.jpg")
    _write_jpg(maria / "1.jpg")

    app = MagicMock()
    app.get.side_effect = [
        [
            SimpleNamespace(
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=np.array([0, 0, 5, 5], dtype=np.float32),
                det_score=0.9,
            )
        ],
        [
            SimpleNamespace(
                embedding=np.array([0.0, 1.0], dtype=np.float32),
                bbox=np.array([0, 0, 6, 6], dtype=np.float32),
                det_score=0.8,
            )
        ],
    ]
    gallery, report = FaceGallery.enroll(tmp_path, ArcFaceExtractor(app=app))
    assert report.enrolled == 2
    assert {record.identity for record in gallery.records} == {"joao", "maria"}

    index_path = tmp_path / "gallery.npz"
    gallery.save(index_path)
    loaded = FaceGallery.load(index_path)
    assert len(loaded) == 2


def test_enroll_skips_image_without_faces(tmp_path) -> None:
    person = tmp_path / "ana"
    person.mkdir()
    _write_jpg(person / "sem_rosto.jpg")
    app = MagicMock()
    app.get.return_value = []
    gallery, report = FaceGallery.enroll(tmp_path, ArcFaceExtractor(app=app))
    assert report.enrolled == 0
    assert len(gallery) == 0
    assert any("nenhum rosto" in item for item in report.skipped)


def test_enroll_skips_multiple_faces_in_gallery_image(tmp_path) -> None:
    person = tmp_path / "ana"
    person.mkdir()
    _write_jpg(person / "grupo.jpg")
    app = MagicMock()
    app.get.return_value = [
        SimpleNamespace(
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            bbox=np.array([0, 0, 5, 5], dtype=np.float32),
            det_score=0.9,
        ),
        SimpleNamespace(
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            bbox=np.array([6, 0, 10, 5], dtype=np.float32),
            det_score=0.8,
        ),
    ]
    gallery, report = FaceGallery.enroll(tmp_path, ArcFaceExtractor(app=app))
    assert report.enrolled == 0
    assert any("rostos" in item for item in report.skipped)

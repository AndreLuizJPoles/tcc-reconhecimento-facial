"""Testes do CLI enroll/identify com extrator mockado."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.main import run, select_face
from src.models import ArcFaceExtractor, FaceDetection


def _write_jpg(path, image: np.ndarray | None = None) -> None:
    blank = image if image is not None else np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok
    path.write_bytes(encoded.tobytes())


def _detection(values: list[float], score: float = 0.99) -> FaceDetection:
    return FaceDetection(
        embedding=np.array(values, dtype=np.float32),
        bbox=np.array([1, 2, 10, 20], dtype=np.float32),
        det_score=score,
    )


def test_select_face_requires_index_when_multiple() -> None:
    detections = [_detection([1.0, 0.0]), _detection([0.0, 1.0])]
    with pytest.raises(ValueError, match="Informe --face-index"):
        select_face(detections, None)


def test_select_face_rejects_out_of_range_index() -> None:
    detections = [_detection([1.0, 0.0])]
    with pytest.raises(ValueError, match="inválido"):
        select_face(detections, 3)


def test_enroll_and_identify_with_mocked_extractor(tmp_path) -> None:
    gallery_root = tmp_path / "gallery"
    person_dir = gallery_root / "joao"
    person_dir.mkdir(parents=True)
    gallery_image = person_dir / "foto.jpg"
    query_image = tmp_path / "query.jpg"
    _write_jpg(gallery_image)
    _write_jpg(query_image)

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
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=np.array([0, 0, 6, 6], dtype=np.float32),
                det_score=0.8,
            )
        ],
    ]
    extractor = ArcFaceExtractor(app=app)
    index_path = tmp_path / "gallery.npz"
    json_path = tmp_path / "out.json"
    figure_path = tmp_path / "out.png"

    enroll_payload = run(
        [
            "enroll",
            "--gallery",
            str(gallery_root),
            "--output",
            str(index_path),
        ],
        extractor=extractor,
    )
    assert enroll_payload["enrolled"] == 1
    assert index_path.exists()

    identify_payload = run(
        [
            "identify",
            "--image",
            str(query_image),
            "--index",
            str(index_path),
            "--output-json",
            str(json_path),
            "--output-figure",
            str(figure_path),
        ],
        extractor=extractor,
    )
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert identify_payload["matched"] is True
    assert identify_payload["identity"] == "joao"
    assert saved["similarity"] == pytest.approx(1.0)
    assert "embedding" not in saved
    assert figure_path.exists()


def test_identify_reports_not_found_below_threshold(tmp_path) -> None:
    gallery_root = tmp_path / "gallery"
    person_dir = gallery_root / "maria"
    person_dir.mkdir(parents=True)
    gallery_image = person_dir / "foto.jpg"
    query_image = tmp_path / "query.jpg"
    _write_jpg(gallery_image)
    _write_jpg(query_image)

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
    extractor = ArcFaceExtractor(app=app)
    index_path = tmp_path / "gallery.npz"

    run(
        ["enroll", "--gallery", str(gallery_root), "--output", str(index_path)],
        extractor=extractor,
    )
    payload = run(
        [
            "identify",
            "--image",
            str(query_image),
            "--index",
            str(index_path),
            "--threshold",
            "0.5",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-figure",
            str(tmp_path / "out.png"),
        ],
        extractor=extractor,
    )
    assert payload["matched"] is False
    assert payload["identity"] is None
    assert payload["similarity"] == pytest.approx(0.0)

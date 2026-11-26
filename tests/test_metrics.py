"""Testes de métricas FP/FN e calibração com vetores sintéticos."""

import numpy as np
import pytest

from src.gallery import FaceGallery, GalleryRecord
from src.metrics import (
    append_metrics_table_row,
    calibrate_threshold,
    collect_calibration_scores,
    confusion_from_decisions,
    evaluate_identification,
)
from src.models import FaceDetection


def test_confusion_from_decisions_counts() -> None:
    counts = confusion_from_decisions(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert counts.tp == 1
    assert counts.fn == 1
    assert counts.fp == 1
    assert counts.tn == 1
    assert counts.precision() == pytest.approx(0.5)
    assert counts.recall() == pytest.approx(0.5)
    assert counts.f1() == pytest.approx(0.5)


def test_calibrate_threshold_prefers_separating_score() -> None:
    scores = [0.9, 0.85, 0.1, 0.05]
    genuine = [True, True, False, False]
    threshold, counts = calibrate_threshold(scores, genuine, metric="f1")
    assert 0.1 < threshold < 0.85
    assert counts.tp == 2
    assert counts.tn == 2
    assert counts.fp == 0
    assert counts.fn == 0


def test_evaluate_identification_with_mock_extractor(tmp_path) -> None:
    class MockExtractor:
        name = "mock"

        def __init__(self, mapping: dict[str, np.ndarray]) -> None:
            self.mapping = mapping

        def extract(self, image: np.ndarray) -> list[FaceDetection]:
            # imagem não importa; usamos chave via atributo injetado no teste
            key = getattr(self, "_next_key")
            vector = self.mapping[key]
            return [
                FaceDetection(
                    embedding=vector,
                    bbox=np.array([0, 0, 5, 5], dtype=np.float32),
                    det_score=0.99,
                )
            ]

    joao = np.array([1.0, 0.0], dtype=np.float32)
    maria = np.array([0.0, 1.0], dtype=np.float32)
    gallery = FaceGallery(
        [
            GalleryRecord(
                identity="joao",
                image_path="joao.jpg",
                embedding=joao,
                bbox=np.array([0, 0, 1, 1], dtype=np.float32),
                det_score=0.9,
            ),
            GalleryRecord(
                identity="maria",
                image_path="maria.jpg",
                embedding=maria,
                bbox=np.array([0, 0, 1, 1], dtype=np.float32),
                det_score=0.9,
            ),
        ]
    )

    import cv2

    q1 = tmp_path / "q_joao.jpg"
    q2 = tmp_path / "q_out.jpg"
    for path in (q1, q2):
        ok, encoded = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        assert ok
        path.write_bytes(encoded.tobytes())

    extractor = MockExtractor({"joao": joao, "out": maria * 0 + np.array([0.0, -1.0], dtype=np.float32)})
    queries = [
        {"identity": "joao", "absolute_path": str(q1), "in_gallery": True},
        {"identity": "estranho", "absolute_path": str(q2), "in_gallery": False},
    ]

    # Força chaves na ordem das queries
    results = []
    for item, key in zip(queries, ["joao", "out"], strict=True):
        extractor._next_key = key
        report = evaluate_identification(
            gallery,
            [item],
            extractor,
            threshold=0.5,
            mode="baseline",
        )
        results.append(report)

    assert results[0].closed_set_accuracy == pytest.approx(1.0)
    assert results[1].open_set_rejection_rate == pytest.approx(1.0)


def test_collect_calibration_scores_includes_impostors_from_other_ids(tmp_path) -> None:
    import cv2

    joao = np.array([1.0, 0.0], dtype=np.float32)
    maria = np.array([0.0, 1.0], dtype=np.float32)
    gallery = FaceGallery(
        [
            GalleryRecord("joao", "joao.jpg", joao, np.array([0, 0, 1, 1], dtype=np.float32), 0.9),
            GalleryRecord("maria", "maria.jpg", maria, np.array([0, 0, 1, 1], dtype=np.float32), 0.9),
        ]
    )

    query_path = tmp_path / "q.jpg"
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    query_path.write_bytes(encoded.tobytes())

    class SingleExtractor:
        name = "mock"

        def extract(self, image: np.ndarray) -> list[FaceDetection]:
            return [FaceDetection(joao, np.array([0, 0, 1, 1], dtype=np.float32), 0.99)]

    scores, genuine = collect_calibration_scores(
        gallery,
        [{"identity": "joao", "absolute_path": str(query_path), "in_gallery": True}],
        SingleExtractor(),
    )
    assert genuine == [True, False]
    threshold, _ = calibrate_threshold(scores, genuine, metric="f1")
    assert 0.0 < threshold < 1.0


def test_append_metrics_table_row_writes_header_on_empty_file(tmp_path) -> None:
    from src.metrics import ConfusionCounts, EvaluationReport

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("", encoding="utf-8")
    report = EvaluationReport(
        mode="baseline",
        threshold=0.5,
        threshold_source="cli",
        confusion=ConfusionCounts(tp=1, fp=0, tn=1, fn=0),
        identification_rate=1.0,
        closed_set_accuracy=1.0,
        open_set_rejection_rate=1.0,
        rank1=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        mean_extract_ms=10.0,
        mean_identify_ms=1.0,
        n_queries=2,
        n_gallery=2,
    )
    append_metrics_table_row(report, csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("mode,threshold,")
    assert len(lines) == 2


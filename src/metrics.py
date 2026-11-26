"""Métricas de identificação 1:N, matriz FP/FN e latência."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.evaluation import FaceComparator
from src.gallery import FaceGallery, GalleryRecord
from src.embedders.base import FaceEmbedder
from src.preprocessing import ImageLoadError, load_image


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else 0.0

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        denom = precision + recall
        return float(2 * precision * recall / denom) if denom else 0.0

    def accuracy(self) -> float:
        return float((self.tp + self.tn) / self.total) if self.total else 0.0


@dataclass
class QueryEvalRow:
    query_path: str
    true_identity: str
    in_gallery: bool
    predicted_identity: str | None
    matched: bool
    similarity: float
    correct: bool
    latency_extract_ms: float
    latency_identify_ms: float


@dataclass
class EvaluationReport:
    mode: str
    threshold: float
    threshold_source: str
    confusion: ConfusionCounts
    identification_rate: float
    closed_set_accuracy: float
    open_set_rejection_rate: float
    rank1: float
    precision: float
    recall: float
    f1: float
    mean_extract_ms: float
    mean_identify_ms: float
    n_queries: int
    n_gallery: int
    rows: list[QueryEvalRow] = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "confusion": asdict(self.confusion),
            "identification_rate": self.identification_rate,
            "closed_set_accuracy": self.closed_set_accuracy,
            "open_set_rejection_rate": self.open_set_rejection_rate,
            "rank1": self.rank1,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_extract_ms": self.mean_extract_ms,
            "mean_identify_ms": self.mean_identify_ms,
            "n_queries": self.n_queries,
            "n_gallery": self.n_gallery,
        }


def confusion_from_decisions(
    y_true_match: Iterable[bool],
    y_pred_match: Iterable[bool],
) -> ConfusionCounts:
    """TP/FP/TN/FN no critério match vs não-match (open-set friendly)."""
    counts = ConfusionCounts()
    for truth, pred in zip(y_true_match, y_pred_match, strict=True):
        if truth and pred:
            counts.tp += 1
        elif not truth and pred:
            counts.fp += 1
        elif not truth and not pred:
            counts.tn += 1
        else:
            counts.fn += 1
    return counts


def calibrate_threshold(
    scores: list[float],
    is_genuine: list[bool],
    *,
    metric: str = "f1",
    grid_size: int = 101,
) -> tuple[float, ConfusionCounts]:
    """Escolhe threshold na validação (maximiza F1 por padrão)."""
    if not scores or len(scores) != len(is_genuine):
        raise ValueError("scores e is_genuine precisam ter o mesmo tamanho não vazio.")

    best_threshold = 0.5
    best_counts = ConfusionCounts()
    best_value = -1.0

    for threshold in np.linspace(-1.0, 1.0, grid_size):
        preds = [score >= threshold for score in scores]
        counts = confusion_from_decisions(is_genuine, preds)
        if metric == "f1":
            value = counts.f1()
        elif metric == "accuracy":
            value = counts.accuracy()
        else:
            raise ValueError(f"Métrica de calibração desconhecida: {metric}")
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_counts = counts

    return best_threshold, best_counts


def collect_calibration_scores(
    gallery: FaceGallery,
    queries: list[dict],
    extractor: FaceEmbedder,
) -> tuple[list[float], list[bool]]:
    """Scores para calibrar threshold: genuine = similaridade com a id correta;
    impostor = melhor similaridade na galeria (query fora da galeria).
    """
    comparator = FaceComparator(threshold=-1.0)
    scores: list[float] = []
    genuine_flags: list[bool] = []

    for item in queries:
        path = item.get("absolute_path") or item["path"]
        true_identity = str(item["identity"])
        in_gallery = bool(item.get("in_gallery", True))
        try:
            image = load_image(path)
            detections = extractor.extract(image)
        except (ImageLoadError, ValueError):
            continue
        if not detections:
            continue
        detection = max(detections, key=lambda face: face.det_score)
        ranked = comparator.identify(detection.embedding, gallery.records, top_k=5)

        if in_gallery:
            same_id = [
                record for record in gallery.records if record.identity == true_identity
            ]
            if not same_id:
                continue
            genuine_score = max(
                comparator.cosine_similarity(detection.embedding, record.embedding)
                for record in same_id
            )
            scores.append(float(genuine_score))
            genuine_flags.append(True)

            other_ids = [
                record for record in gallery.records if record.identity != true_identity
            ]
            if other_ids:
                impostor_score = max(
                    comparator.cosine_similarity(detection.embedding, record.embedding)
                    for record in other_ids
                )
                scores.append(float(impostor_score))
                genuine_flags.append(False)
        else:
            scores.append(float(ranked.similarity))
            genuine_flags.append(False)

    return scores, genuine_flags


def evaluate_identification(
    gallery: FaceGallery,
    queries: list[dict],
    extractor: FaceEmbedder,
    *,
    threshold: float,
    mode: str = "baseline",
    threshold_source: str = "val",
) -> EvaluationReport:
    """Avalia 1:N com threshold fixo (calibrado na val, aplicado no test)."""
    ranker = FaceComparator(threshold=-1.0)
    rows: list[QueryEvalRow] = []
    extract_times: list[float] = []
    identify_times: list[float] = []

    y_true: list[bool] = []
    y_pred: list[bool] = []
    closed_correct = 0
    closed_total = 0
    open_reject = 0
    open_total = 0
    rank1_hits = 0
    closed_for_rank = 0

    for item in queries:
        path = item.get("absolute_path") or item["path"]
        true_identity = str(item["identity"])
        in_gallery = bool(item.get("in_gallery", True))

        try:
            t0 = time.perf_counter()
            image = load_image(path)
            detections = extractor.extract(image)
            extract_ms = (time.perf_counter() - t0) * 1000.0
        except (ImageLoadError, ValueError):
            continue
        if not detections:
            continue

        detection = max(detections, key=lambda face: face.det_score)
        t1 = time.perf_counter()
        ranked = ranker.identify(detection.embedding, gallery.records, top_k=5)
        identify_ms = (time.perf_counter() - t1) * 1000.0
        extract_times.append(extract_ms)
        identify_times.append(identify_ms)

        best_identity = ranked.candidates[0].identity if ranked.candidates else None
        best_similarity = float(ranked.similarity)
        matched = best_similarity >= threshold
        predicted = best_identity if matched else None

        if in_gallery:
            closed_total += 1
            closed_for_rank += 1
            if best_identity == true_identity:
                rank1_hits += 1
            correct = predicted == true_identity
            if correct:
                closed_correct += 1
            y_true.append(True)
            y_pred.append(predicted == true_identity)
        else:
            open_total += 1
            correct = not matched
            if not matched:
                open_reject += 1
            y_true.append(False)
            y_pred.append(matched)

        rows.append(
            QueryEvalRow(
                query_path=str(path),
                true_identity=true_identity,
                in_gallery=in_gallery,
                predicted_identity=predicted,
                matched=matched,
                similarity=best_similarity,
                correct=correct,
                latency_extract_ms=extract_ms,
                latency_identify_ms=identify_ms,
            )
        )

    confusion = confusion_from_decisions(y_true, y_pred)
    n_queries = len(rows)
    identification_rate = (
        float(sum(1 for row in rows if row.correct) / n_queries) if n_queries else 0.0
    )

    return EvaluationReport(
        mode=mode,
        threshold=float(threshold),
        threshold_source=threshold_source,
        confusion=confusion,
        identification_rate=identification_rate,
        closed_set_accuracy=(
            float(closed_correct / closed_total) if closed_total else 0.0
        ),
        open_set_rejection_rate=(
            float(open_reject / open_total) if open_total else 0.0
        ),
        rank1=float(rank1_hits / closed_for_rank) if closed_for_rank else 0.0,
        precision=confusion.precision(),
        recall=confusion.recall(),
        f1=confusion.f1(),
        mean_extract_ms=float(np.mean(extract_times)) if extract_times else 0.0,
        mean_identify_ms=float(np.mean(identify_times)) if identify_times else 0.0,
        n_queries=n_queries,
        n_gallery=len(gallery.records),
        rows=rows,
    )


def enroll_from_split_items(
    items: list[dict],
    extractor: FaceEmbedder,
) -> FaceGallery:
    """Monta galeria a partir de itens do splits.json."""
    records: list[GalleryRecord] = []
    for item in items:
        path = item.get("absolute_path") or item["path"]
        try:
            image = load_image(path)
            detections = extractor.extract(image)
        except (ImageLoadError, ValueError):
            continue
        if len(detections) != 1:
            continue
        detection = detections[0]
        records.append(
            GalleryRecord(
                identity=str(item["identity"]),
                image_path=str(path),
                embedding=detection.embedding,
                bbox=detection.bbox,
                det_score=detection.det_score,
            )
        )
    return FaceGallery(records)


def save_metrics_csv(report: EvaluationReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_path",
                "true_identity",
                "in_gallery",
                "predicted_identity",
                "matched",
                "similarity",
                "correct",
                "latency_extract_ms",
                "latency_identify_ms",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(asdict(row))
    return output


def save_metrics_summary(report: EvaluationReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_summary_dict(), indent=2), encoding="utf-8")
    return output


def append_metrics_table_row(
    report: EvaluationReport,
    csv_path: str | Path,
) -> Path:
    """Acrescenta uma linha resumo em results/metrics.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "threshold",
        "threshold_source",
        "identification_rate",
        "closed_set_accuracy",
        "open_set_rejection_rate",
        "rank1",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "mean_extract_ms",
        "mean_identify_ms",
        "n_queries",
        "n_gallery",
    ]
    row = {
        "mode": report.mode,
        "threshold": report.threshold,
        "threshold_source": report.threshold_source,
        "identification_rate": report.identification_rate,
        "closed_set_accuracy": report.closed_set_accuracy,
        "open_set_rejection_rate": report.open_set_rejection_rate,
        "rank1": report.rank1,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "tp": report.confusion.tp,
        "fp": report.confusion.fp,
        "tn": report.confusion.tn,
        "fn": report.confusion.fn,
        "mean_extract_ms": report.mean_extract_ms,
        "mean_identify_ms": report.mean_identify_ms,
        "n_queries": report.n_queries,
        "n_gallery": report.n_gallery,
    }
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path

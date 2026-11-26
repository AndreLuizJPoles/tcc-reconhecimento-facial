"""Visualização da consulta contra a galeria (match ou não encontrado)."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import IdentificationResult
from src.models import FaceDetection


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_boxes(image: np.ndarray, detections: list[FaceDetection], selected_index: int) -> np.ndarray:
    canvas = image.copy()
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox.astype(int)
        color = (0, 255, 0) if index == selected_index else (0, 165, 255)
        thickness = 3 if index == selected_index else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{index} ({detection.det_score:.2f})"
        cv2.putText(
            canvas,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_identification_figure(
    query_image: np.ndarray,
    query_detections: list[FaceDetection],
    query_index: int,
    identification: IdentificationResult,
    output_path: str | Path,
    gallery_image: np.ndarray | None = None,
    gallery_detections: list[FaceDetection] | None = None,
) -> Path:
    annotated_query = _draw_boxes(query_image, query_detections, query_index)
    if identification.matched:
        title = (
            f"Match: {identification.identity} | "
            f"Similaridade: {identification.similarity:.4f} | "
            f"Threshold: {identification.threshold:.2f}"
        )
    else:
        title = (
            f"Não encontrado | Melhor similaridade: {identification.similarity:.4f} | "
            f"Threshold: {identification.threshold:.2f}"
        )

    if identification.matched and gallery_image is not None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(_bgr_to_rgb(annotated_query))
        axes[0].set_title("Consulta")
        axes[0].axis("off")
        annotated_gallery = gallery_image
        if gallery_detections:
            annotated_gallery = _draw_boxes(gallery_image, gallery_detections, 0)
        axes[1].imshow(_bgr_to_rgb(annotated_gallery))
        axes[1].set_title(f"Galeria: {identification.identity}")
        axes[1].axis("off")
    else:
        figure, axis = plt.subplots(1, 1, figsize=(7, 6))
        axis.imshow(_bgr_to_rgb(annotated_query))
        axis.set_title("Consulta")
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return output

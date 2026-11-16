"""Extração de embeddings faciais com ArcFace (InsightFace buffalo_l)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.preprocessing import validate_image_array


@dataclass
class FaceDetection:
    embedding: np.ndarray
    bbox: np.ndarray
    det_score: float


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    if embedding.size == 0:
        raise ValueError("O embedding está vazio.")
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Não é possível normalizar um embedding nulo.")
    return embedding / norm


class ArcFaceExtractor:
    """Carrega buffalo_l uma vez e extrai embeddings ArcFace alinhados pelo pipeline."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
        app: Any | None = None,
    ) -> None:
        if app is not None:
            self._app = app
            return

        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        validated = validate_image_array(image)
        faces = self._app.get(validated)
        detections: list[FaceDetection] = []

        for face in faces or []:
            detections.append(
                FaceDetection(
                    embedding=self._normalized_embedding(face),
                    bbox=np.asarray(face.bbox, dtype=np.float32),
                    det_score=float(getattr(face, "det_score", 0.0)),
                )
            )

        return detections

    def _normalized_embedding(self, face: object) -> np.ndarray:
        normed = getattr(face, "normed_embedding", None)
        if normed is not None:
            return l2_normalize(normed)

        embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ValueError("A detecção não contém embedding facial.")
        return l2_normalize(embedding)

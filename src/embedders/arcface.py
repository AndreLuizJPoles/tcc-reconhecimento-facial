"""ArcFace via InsightFace buffalo_l (detecção + alinhamento + embedding)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models import FaceDetection, l2_normalize
from src.preprocessing import validate_image_array


class ArcFaceExtractor:
    """Carrega buffalo_l uma vez e extrai embeddings ArcFace alinhados.

    ROI: o detector do InsightFace devolve a bbox da face. O alinhamento
    pelos landmarks ocorre antes do reconhecimento. A avaliação e o cadastro
    usam sempre essa face alinhada, nunca a imagem crua inteira.
    """

    name = "arcface"

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

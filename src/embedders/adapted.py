"""ArcFace + projeção linear treinável (backbone ONNX congelado)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embedders.arcface import ArcFaceExtractor
from src.finetuning import LinearAdapter, apply_adapter
from src.models import FaceDetection


class AdaptedArcFaceExtractor:
    """Mesma detecção/alinhamento InsightFace; embedding passa pelo LinearAdapter.

    O buffalo_l permanece congelado (ONNX). Só a projeção linear é treinável
    e especializa o espaço ao subconjunto LFW usado no treino do adaptador.
    """

    name = "arcface-adapted"

    def __init__(
        self,
        base: ArcFaceExtractor | None = None,
        adapter: LinearAdapter | None = None,
        device: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        if adapter is None:
            raise ValueError("AdaptedArcFaceExtractor exige um LinearAdapter carregado.")
        self._base = base if base is not None else ArcFaceExtractor(**base_kwargs)
        self._adapter = adapter
        self._device = device

    @property
    def adapter(self) -> LinearAdapter:
        return self._adapter

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        detections = self._base.extract(image)
        adapted: list[FaceDetection] = []
        for detection in detections:
            embedding = apply_adapter(
                self._adapter,
                detection.embedding,
                device=self._device,
            )
            adapted.append(
                FaceDetection(
                    embedding=embedding,
                    bbox=detection.bbox,
                    det_score=detection.det_score,
                )
            )
        return adapted

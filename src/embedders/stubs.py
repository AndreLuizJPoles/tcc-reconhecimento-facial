"""Placeholders para outros métodos do TCC (não implementados nesta etapa)."""

from __future__ import annotations

import numpy as np

from src.models import FaceDetection


class CosFaceExtractor:
    """Stub: CosFace fica para etapa futura do grupo."""

    name = "cosface"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "CosFace ainda não está implementado. Use --model arcface."
        )

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        raise NotImplementedError(
            "CosFace ainda não está implementado. Use --model arcface."
        )


class AuraFaceExtractor:
    """Stub: AuraFace / uso comercial fica fora do escopo desta entrega."""

    name = "auraface"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "AuraFace ainda não está implementado. Use --model arcface."
        )

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        raise NotImplementedError(
            "AuraFace ainda não está implementado. Use --model arcface."
        )

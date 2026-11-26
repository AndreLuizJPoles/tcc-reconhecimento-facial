"""Protocolo comum de extrator facial (detecta ROI, alinha e embute)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from src.models import FaceDetection


@runtime_checkable
class FaceEmbedder(Protocol):
    """Extrai faces alinhadas (ROI) e embeddings L2-normalizados.

    O fluxo esperado é: detectar bbox (ROI) -> alinhar -> embedding.
    No ArcFace/InsightFace isso ocorre dentro do pipeline buffalo_l.
    """

    name: str

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        """Detecta rostos na imagem e devolve embeddings por face."""
        ...

"""Tipos compartilhados e factory de extratores faciais."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.embedders.base import FaceEmbedder


@dataclass
class FaceDetection:
    """Uma face detectada: embedding L2, bbox (ROI) e score do detector."""

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


SUPPORTED_MODELS = ("arcface", "cosface", "auraface")
IMPLEMENTED_MODELS = ("arcface",)


def get_embedder(
    model_name: str = "arcface",
    adapter: str | Path | None = None,
    ctx_id: int = -1,
    det_size: tuple[int, int] = (640, 640),
    app: Any | None = None,
    **kwargs: Any,
) -> FaceEmbedder:
    """Factory de extratores. Só ArcFace está implementado de fato.

    adapter:
      - None / "none" / "" → buffalo_l puro
      - caminho .pt → AdaptedArcFaceExtractor com LinearAdapter
    """
    from src.embedders.arcface import ArcFaceExtractor

    name = (model_name or "arcface").strip().lower()
    adapter_arg = None if adapter in (None, "", "none") else Path(adapter)

    if name == "cosface":
        from src.embedders.stubs import CosFaceExtractor

        return CosFaceExtractor(ctx_id=ctx_id, det_size=det_size, **kwargs)

    if name == "auraface":
        from src.embedders.stubs import AuraFaceExtractor

        return AuraFaceExtractor(ctx_id=ctx_id, det_size=det_size, **kwargs)

    if name != "arcface":
        raise ValueError(
            f"Modelo desconhecido: {model_name!r}. "
            f"Suportados: {', '.join(SUPPORTED_MODELS)} "
            f"(implementado: {', '.join(IMPLEMENTED_MODELS)})."
        )

    buffalo_name = kwargs.pop("buffalo_name", "buffalo_l")
    base = ArcFaceExtractor(
        model_name=buffalo_name,
        ctx_id=ctx_id,
        det_size=det_size,
        app=app,
        **kwargs,
    )
    if adapter_arg is None:
        return base

    from src.embedders.adapted import AdaptedArcFaceExtractor
    from src.finetuning import load_adapter

    linear_adapter = load_adapter(adapter_arg)
    return AdaptedArcFaceExtractor(base=base, adapter=linear_adapter)


def __getattr__(name: str) -> Any:
    """Lazy exports para evitar import circular com src.embedders.*."""
    if name == "ArcFaceExtractor":
        from src.embedders.arcface import ArcFaceExtractor

        return ArcFaceExtractor
    if name == "FaceEmbedder":
        from src.embedders.base import FaceEmbedder

        return FaceEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

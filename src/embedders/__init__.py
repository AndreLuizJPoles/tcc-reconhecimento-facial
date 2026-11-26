"""Extratores de embedding facial (ArcFace implementado; demais só stubs)."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ArcFaceExtractor":
        from src.embedders.arcface import ArcFaceExtractor

        return ArcFaceExtractor
    if name == "AdaptedArcFaceExtractor":
        from src.embedders.adapted import AdaptedArcFaceExtractor

        return AdaptedArcFaceExtractor
    if name == "FaceEmbedder":
        from src.embedders.base import FaceEmbedder

        return FaceEmbedder
    if name == "CosFaceExtractor":
        from src.embedders.stubs import CosFaceExtractor

        return CosFaceExtractor
    if name == "AuraFaceExtractor":
        from src.embedders.stubs import AuraFaceExtractor

        return AuraFaceExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdaptedArcFaceExtractor",
    "ArcFaceExtractor",
    "AuraFaceExtractor",
    "CosFaceExtractor",
    "FaceEmbedder",
]

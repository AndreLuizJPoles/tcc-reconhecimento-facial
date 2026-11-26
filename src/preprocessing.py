"""Carregamento e validação de imagens para o pipeline ArcFace."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(ValueError):
    """Imagem inexistente, ilegível ou inválida."""


def validate_image_array(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise ImageLoadError("A imagem não foi carregada (valor nulo).")
    if not isinstance(image, np.ndarray):
        raise ImageLoadError("A entrada precisa ser um array NumPy.")
    if image.size == 0:
        raise ImageLoadError("A imagem está vazia.")
    if image.ndim not in (2, 3):
        raise ImageLoadError("A imagem precisa ter 2 ou 3 dimensões.")
    return image


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f"Arquivo inexistente: {path}")
    if not path.is_file():
        raise ImageLoadError(f"O caminho não é um arquivo: {path}")

    file_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Não foi possível ler a imagem: {path}")
    return validate_image_array(image)

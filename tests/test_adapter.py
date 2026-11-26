"""Testes do LinearAdapter (shape 512→512 e L2)."""

import numpy as np
import pytest
import torch

from src.embedders.adapted import AdaptedArcFaceExtractor
from src.embedders.arcface import ArcFaceExtractor
from src.finetuning import (
    LinearAdapter,
    apply_adapter,
    load_adapter,
    save_adapter,
    train_linear_adapter,
)
from src.models import FaceDetection, get_embedder


def test_linear_adapter_forward_shape_and_l2() -> None:
    adapter = LinearAdapter(dim=512)
    batch = torch.randn(4, 512)
    out = adapter(batch)
    assert out.shape == (4, 512)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_apply_adapter_numpy_roundtrip(tmp_path) -> None:
    adapter = LinearAdapter(dim=8)
    checkpoint = save_adapter(adapter, tmp_path / "a.pt")
    loaded = load_adapter(checkpoint, map_location="cpu")
    vector = np.random.randn(8).astype(np.float32)
    adapted = apply_adapter(loaded, vector, device="cpu")
    assert adapted.shape == (8,)
    assert adapted.dtype == np.float32
    assert np.isclose(np.linalg.norm(adapted), 1.0, atol=1e-5)


def test_train_linear_adapter_on_synthetic_embeddings(tmp_path) -> None:
    rng = np.random.default_rng(0)
    embeddings = []
    labels = []
    for identity_index in range(3):
        center = rng.normal(size=16).astype(np.float32)
        center /= np.linalg.norm(center)
        for _ in range(5):
            noise = rng.normal(scale=0.05, size=16).astype(np.float32)
            vector = center + noise
            vector /= np.linalg.norm(vector)
            embeddings.append(vector)
            labels.append(f"id_{identity_index}")

    report = train_linear_adapter(
        np.stack(embeddings),
        labels,
        epochs=3,
        batch_size=8,
        device="cpu",
        output_path=tmp_path / "adapter.pt",
        seed=0,
    )
    assert report.n_identities == 3
    assert report.n_samples == 15
    assert (tmp_path / "adapter.pt").exists()


def test_adapted_extractor_applies_projection() -> None:
    class FakeBase(ArcFaceExtractor):
        def __init__(self) -> None:
            self._app = None

        def extract(self, image: np.ndarray) -> list[FaceDetection]:
            return [
                FaceDetection(
                    embedding=np.ones(8, dtype=np.float32) / np.sqrt(8),
                    bbox=np.array([0, 0, 1, 1], dtype=np.float32),
                    det_score=0.9,
                )
            ]

    adapter = LinearAdapter(dim=8)
    adapted = AdaptedArcFaceExtractor(base=FakeBase(), adapter=adapter, device="cpu")
    assert adapted.name == "arcface-adapted"
    detections = adapted.extract(np.zeros((8, 8, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].embedding.shape == (8,)
    assert np.isclose(np.linalg.norm(detections[0].embedding), 1.0, atol=1e-5)


def test_get_embedder_with_adapter_checkpoint(tmp_path) -> None:
    adapter = LinearAdapter(dim=512)
    path = save_adapter(adapter, tmp_path / "arcface_linear.pt")
    from unittest.mock import MagicMock

    app = MagicMock()
    app.get.return_value = []
    embedder = get_embedder(model_name="arcface", adapter=path, app=app)
    assert embedder.name == "arcface-adapted"

"""Adaptador leve sobre embeddings ArcFace congelados (buffalo_l / ONNX).

O backbone InsightFace não é treinado (ONNX). Aqui só uma projeção linear
512→512 com normalização L2 é otimizada com cross-entropy sobre identidades
do subconjunto LFW de treino.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ADAPTER_PATH = Path("results/adapters/arcface_linear.pt")
DEFAULT_EMBEDDING_DIM = 512


class LinearAdapter:
    """Projeção linear + L2. Interface fina sobre torch.nn quando disponível."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM, module: Any | None = None) -> None:
        self.dim = int(dim)
        if module is not None:
            self._module = module
            return

        import torch
        from torch import nn

        self._module = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=True),
        )

    @property
    def module(self) -> Any:
        return self._module

    def forward(self, embeddings: Any) -> Any:
        import torch.nn.functional as F

        projected = self._module(embeddings)
        return F.normalize(projected, p=2, dim=-1)

    def __call__(self, embeddings: Any) -> Any:
        return self.forward(embeddings)

    def state_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "adapter": self._module.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "dim" in state:
            self.dim = int(state["dim"])
        weights = state.get("adapter", state)
        self._module.load_state_dict(weights)

    def to(self, device: str | Any) -> LinearAdapter:
        self._module.to(device)
        return self

    def eval(self) -> LinearAdapter:
        self._module.eval()
        return self

    def train(self, mode: bool = True) -> LinearAdapter:
        self._module.train(mode)
        return self

    def parameters(self):
        return self._module.parameters()


def apply_adapter(
    adapter: LinearAdapter,
    embedding: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Aplica o adaptador a um embedding NumPy e devolve vetor L2 float32."""
    import torch

    vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if vector.shape[1] != adapter.dim:
        raise ValueError(
            f"Dimensão do embedding {vector.shape[1]} difere do adaptador {adapter.dim}."
        )

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter.eval()
    tensor = torch.from_numpy(vector).to(resolved)
    with torch.no_grad():
        adapted = adapter(tensor).detach().cpu().numpy().reshape(-1)
    return np.asarray(adapted, dtype=np.float32)


def save_adapter(
    adapter: LinearAdapter,
    output_path: str | Path,
    meta: dict[str, Any] | None = None,
) -> Path:
    import torch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.state_dict()
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)
    return path


def load_adapter(
    checkpoint_path: str | Path,
    map_location: str | None = None,
) -> LinearAdapter:
    import torch

    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Checkpoint do adaptador inexistente: {path}")

    location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(path, map_location=location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=location)

    dim = int(state.get("dim", DEFAULT_EMBEDDING_DIM))
    adapter = LinearAdapter(dim=dim)
    adapter.load_state_dict(state)
    adapter.to(location)
    adapter.eval()
    return adapter


@dataclass
class AdapterTrainReport:
    epochs: int
    final_loss: float
    n_identities: int
    n_samples: int
    device: str
    checkpoint: str
    identity_to_index: dict[str, int]


def _resolve_device(device: str | None) -> str:
    import torch

    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def extract_frozen_embeddings(
    image_paths: list[str | Path],
    identities: list[str],
    extractor: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extrai embeddings congelados; ignora imagens sem face única."""
    from src.preprocessing import ImageLoadError, load_image

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    kept_paths: list[str] = []
    skipped: list[str] = []

    for image_path, identity in zip(image_paths, identities, strict=True):
        path = Path(image_path)
        try:
            image = load_image(path)
            detections = extractor.extract(image)
        except (ImageLoadError, ValueError) as error:
            skipped.append(f"{path}: {error}")
            continue

        if not detections:
            skipped.append(f"{path}: nenhum rosto detectado")
            continue
        if len(detections) > 1:
            skipped.append(f"{path}: {len(detections)} rostos")
            continue

        embeddings.append(np.asarray(detections[0].embedding, dtype=np.float32))
        labels.append(identity)
        kept_paths.append(str(path))

    if not embeddings:
        raise ValueError("Nenhum embedding válido para treinar o adaptador.")

    return np.stack(embeddings), np.array(labels), kept_paths, skipped


def train_linear_adapter(
    embeddings: np.ndarray,
    identity_labels: np.ndarray | list[str],
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
    output_path: str | Path = DEFAULT_ADAPTER_PATH,
    seed: int = 42,
) -> AdapterTrainReport:
    """Treina LinearAdapter + cabeça linear com cross-entropy nas identidades."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    resolved_device = _resolve_device(device)
    labels = np.asarray(identity_labels)
    unique = sorted(set(str(label) for label in labels))
    identity_to_index = {name: index for index, name in enumerate(unique)}
    y = np.array([identity_to_index[str(label)] for label in labels], dtype=np.int64)

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings precisa ser uma matriz (N, D).")
    dim = int(matrix.shape[1])

    torch.manual_seed(seed)
    adapter = LinearAdapter(dim=dim).to(resolved_device)
    classifier = nn.Linear(dim, len(unique)).to(resolved_device)
    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    x_tensor = torch.from_numpy(matrix)
    y_tensor = torch.from_numpy(y)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=min(batch_size, len(matrix)),
        shuffle=True,
    )

    adapter.train(True)
    classifier.train(True)
    final_loss = 0.0

    for _epoch in range(max(1, epochs)):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad()
            adapted = adapter(batch_x)
            logits = classifier(adapted)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    adapter.eval()
    checkpoint = save_adapter(
        adapter,
        output_path,
        meta={
            "n_identities": len(unique),
            "n_samples": int(matrix.shape[0]),
            "epochs": int(epochs),
            "final_loss": final_loss,
            "device": resolved_device,
            "loss": "cross_entropy",
            "identity_to_index": identity_to_index,
        },
    )

    meta_path = Path(checkpoint).with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "n_identities": len(unique),
                "n_samples": int(matrix.shape[0]),
                "epochs": int(epochs),
                "final_loss": final_loss,
                "device": resolved_device,
                "loss": "cross_entropy",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return AdapterTrainReport(
        epochs=int(epochs),
        final_loss=final_loss,
        n_identities=len(unique),
        n_samples=int(matrix.shape[0]),
        device=resolved_device,
        checkpoint=str(checkpoint),
        identity_to_index=identity_to_index,
    )

"""Evaluate e train-adapter com mocks (sem LFW real nem buffalo_l)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.embedders.arcface import ArcFaceExtractor
from src.main import run


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    path.write_bytes(encoded.tobytes())


def _tiny_splits(root: Path) -> Path:
    identities = {
        "alice": 4,
        "bob": 4,
        "carol": 4,
        "dave": 3,
    }
    raw = root / "raw"
    paths: dict[str, list[Path]] = {}
    for name, count in identities.items():
        paths[name] = []
        for index in range(count):
            path = raw / name / f"{name}_{index}.jpg"
            _write_jpg(path)
            paths[name].append(path)

    splits = {
        "dataset": "lfw",
        "seed": 0,
        "min_images": 3,
        "train_adapter": {
            "identities": ["alice", "bob"],
            "images": [
                {
                    "identity": "alice",
                    "path": str(paths["alice"][0].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["alice"][0]),
                },
                {
                    "identity": "alice",
                    "path": str(paths["alice"][1].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["alice"][1]),
                },
                {
                    "identity": "bob",
                    "path": str(paths["bob"][0].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["bob"][0]),
                },
                {
                    "identity": "bob",
                    "path": str(paths["bob"][1].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["bob"][1]),
                },
            ],
        },
        "val": {
            "identities": ["carol"],
            "gallery": [
                {
                    "identity": "carol",
                    "path": str(paths["carol"][0].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["carol"][0]),
                }
            ],
            "queries": [
                {
                    "identity": "carol",
                    "path": str(paths["carol"][1].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["carol"][1]),
                    "in_gallery": True,
                }
            ],
        },
        "test": {
            "identities_in_gallery": ["carol"],
            "identities_open_set": ["dave"],
            "gallery": [
                {
                    "identity": "carol",
                    "path": str(paths["carol"][2].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["carol"][2]),
                }
            ],
            "queries": [
                {
                    "identity": "carol",
                    "path": str(paths["carol"][3].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["carol"][3]),
                    "in_gallery": True,
                },
                {
                    "identity": "dave",
                    "path": str(paths["dave"][0].relative_to(root)).replace("\\", "/"),
                    "absolute_path": str(paths["dave"][0]),
                    "in_gallery": False,
                },
            ],
        },
    }
    splits_path = root / "splits.json"
    import json

    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return splits_path


def _face(embedding: np.ndarray):
    return SimpleNamespace(
        embedding=embedding,
        bbox=np.array([0, 0, 5, 5], dtype=np.float32),
        det_score=0.95,
    )


def test_train_adapter_and_evaluate_with_mocks(tmp_path) -> None:
    root = tmp_path / "lfw"
    splits_path = _tiny_splits(root)
    adapter_path = tmp_path / "adapter.pt"

    # Embeddings determinísticos por identidade (via contador de chamadas)
    identity_vectors = {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "carol": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "dave": np.array([0.7, 0.7, 0.0], dtype=np.float32),
    }

    def side_effect(image):
        # OpenCV decode não carrega identidade; usamos ordem das chamadas + mapa
        # Alternativa: hash do caminho não está disponível. Usamos fila.
        key = side_effect.queue.pop(0)
        return [_face(identity_vectors[key])]

    side_effect.queue = [
        "alice",
        "alice",
        "bob",
        "bob",
        # evaluate val gallery + query
        "carol",
        "carol",
        # evaluate test gallery + queries
        "carol",
        "carol",
        "dave",
    ]

    app = MagicMock()
    app.get.side_effect = side_effect
    extractor = ArcFaceExtractor(app=app)

    train_payload = run(
        [
            "train-adapter",
            "--splits",
            str(splits_path),
            "--output",
            str(adapter_path),
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ],
        extractor=extractor,
    )
    assert adapter_path.exists()
    assert train_payload["n_samples"] == 4

    # evaluate com --threshold fixo: val gallery + test gallery + 2 queries
    side_effect.queue = ["carol", "carol", "carol", "dave"]
    app.get.side_effect = side_effect
    extractor = ArcFaceExtractor(app=app)

    eval_payload = run(
        [
            "evaluate",
            "--splits",
            str(splits_path),
            "--adapter",
            "none",
            "--model",
            "arcface",
            "--metrics-csv",
            str(tmp_path / "metrics.csv"),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-detail-csv",
            str(tmp_path / "detail.csv"),
            "--threshold",
            "0.5",
        ],
        extractor=extractor,
    )
    assert eval_payload["mode"] == "baseline"
    assert eval_payload["n_queries"] >= 1
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()

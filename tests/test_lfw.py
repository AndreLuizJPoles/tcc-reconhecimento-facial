"""Testes do prepare-lfw com filesystem temporário (sem download)."""

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.lfw import build_splits, prepare_lfw, scan_lfw_identities
from src.main import run


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", np.zeros((24, 24, 3), dtype=np.uint8))
    assert ok
    path.write_bytes(encoded.tobytes())


def _make_raw_lfw(raw_dir: Path, n_ids: int = 10, images_per_id: int = 6) -> None:
    for index in range(n_ids):
        identity = f"person_{index:02d}"
        for photo in range(images_per_id):
            _write_jpg(raw_dir / identity / f"{identity}_{photo:04d}.jpg")
    # Identidades escassas para open-set
    for index in range(3):
        identity = f"scarce_{index:02d}"
        _write_jpg(raw_dir / identity / f"{identity}_0001.jpg")


def test_scan_and_build_splits(tmp_path) -> None:
    raw = tmp_path / "raw"
    _make_raw_lfw(raw)
    identities = scan_lfw_identities(raw)
    assert len(identities) == 13
    splits = build_splits(identities, root=tmp_path, min_images=5, seed=0)
    assert splits["train_adapter"]["identities"]
    assert splits["val"]["gallery"]
    assert splits["test"]["gallery"]
    assert splits["test"]["queries"]
    train_ids = set(splits["train_adapter"]["identities"])
    val_ids = set(splits["val"]["identities"])
    test_ids = set(splits["test"]["identities_in_gallery"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_prepare_lfw_without_download(tmp_path) -> None:
    root = tmp_path / "lfw"
    raw = root / "raw"
    _make_raw_lfw(raw)
    report = prepare_lfw(
        root=root,
        download=False,
        min_images=5,
        seed=1,
        materialize=True,
    )
    assert Path(report.splits_path).exists()
    assert (root / "gallery").is_dir()
    assert (root / "query").is_dir()
    splits = json.loads(Path(report.splits_path).read_text(encoding="utf-8"))
    assert "train_adapter" in splits
    assert report.n_gallery_images > 0
    assert report.n_query_images > 0


def test_prepare_lfw_cli(tmp_path) -> None:
    root = tmp_path / "lfw"
    _make_raw_lfw(root / "raw")
    payload = run(
        [
            "prepare-lfw",
            "--root",
            str(root),
            "--no-download",
            "--min-images",
            "5",
            "--seed",
            "2",
        ]
    )
    assert payload["n_train_identities"] >= 1
    assert Path(payload["splits_path"]).exists()


def test_prepare_lfw_from_local_archive(tmp_path) -> None:
    people_root = tmp_path / "lfw"
    _make_raw_lfw(people_root, n_ids=8, images_per_id=5)
    archive_path = tmp_path / "lfw.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(people_root, arcname="lfw")

    root = tmp_path / "prepared"
    report = prepare_lfw(
        root=root,
        download=True,
        archive=archive_path,
        min_images=5,
        seed=3,
        materialize=True,
    )
    assert Path(report.splits_path).exists()
    assert (root / "raw" / "person_00").is_dir()
    assert report.n_identities_raw >= 8

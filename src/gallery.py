"""Cadastro da galeria de identidades a partir de pastas em disco."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.embedders.base import FaceEmbedder
from src.models import FaceDetection
from src.preprocessing import ImageLoadError, load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class GalleryRecord:
    identity: str
    image_path: str
    embedding: np.ndarray
    bbox: np.ndarray
    det_score: float


@dataclass
class EnrollReport:
    enrolled: int
    skipped: list[str] = field(default_factory=list)


class FaceGallery:
    """Galeria 1:N: cada subpasta de data/gallery é uma identidade."""

    def __init__(self, records: list[GalleryRecord] | None = None) -> None:
        self.records = records or []

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def enroll(
        cls,
        gallery_dir: str | Path,
        extractor: FaceEmbedder,
    ) -> tuple[FaceGallery, EnrollReport]:
        root = Path(gallery_dir)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"A pasta da galeria não existe: {root}")

        identity_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if not identity_dirs:
            raise ValueError(
                f"Nenhuma identidade encontrada em {root}. "
                "Use subpastas: data/gallery/<identidade>/foto.jpg"
            )

        records: list[GalleryRecord] = []
        skipped: list[str] = []

        for identity_dir in identity_dirs:
            identity = identity_dir.name
            image_paths = sorted(
                path
                for path in identity_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not image_paths:
                skipped.append(f"{identity}: pasta sem imagens suportadas")
                continue

            for image_path in image_paths:
                try:
                    image = load_image(image_path)
                    detections = extractor.extract(image)
                except (ImageLoadError, ValueError) as error:
                    skipped.append(f"{image_path}: {error}")
                    continue

                if not detections:
                    skipped.append(f"{image_path}: nenhum rosto detectado")
                    continue
                if len(detections) > 1:
                    skipped.append(
                        f"{image_path}: {len(detections)} rostos; "
                        "cadastro ignorado para não escolher a pessoa errada"
                    )
                    continue

                detection = detections[0]
                records.append(
                    GalleryRecord(
                        identity=identity,
                        image_path=str(image_path),
                        embedding=detection.embedding,
                        bbox=detection.bbox,
                        det_score=detection.det_score,
                    )
                )

        report = EnrollReport(enrolled=len(records), skipped=skipped)
        return cls(records), report

    def save(self, output_path: str | Path) -> Path:
        if not self.records:
            raise ValueError("Não há registros para salvar na galeria.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.stack([record.embedding for record in self.records])
        bboxes = np.stack([record.bbox for record in self.records])
        det_scores = np.array(
            [record.det_score for record in self.records],
            dtype=np.float32,
        )
        identities = np.array([record.identity for record in self.records])
        image_paths = np.array([record.image_path for record in self.records])
        np.savez(
            path,
            embeddings=embeddings,
            bboxes=bboxes,
            det_scores=det_scores,
            identities=identities,
            image_paths=image_paths,
        )
        return path

    @classmethod
    def load(cls, index_path: str | Path) -> FaceGallery:
        path = Path(index_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Índice da galeria inexistente: {path}")

        with np.load(path, allow_pickle=True) as data:
            embeddings = data["embeddings"]
            bboxes = data["bboxes"]
            det_scores = data["det_scores"]
            identities = data["identities"]
            image_paths = data["image_paths"]

        records = [
            GalleryRecord(
                identity=str(identities[index]),
                image_path=str(image_paths[index]),
                embedding=np.asarray(embeddings[index], dtype=np.float32),
                bbox=np.asarray(bboxes[index], dtype=np.float32),
                det_score=float(det_scores[index]),
            )
            for index in range(len(identities))
        ]
        return cls(records)

    def as_detection(self, record: GalleryRecord) -> FaceDetection:
        return FaceDetection(
            embedding=record.embedding,
            bbox=record.bbox,
            det_score=record.det_score,
        )

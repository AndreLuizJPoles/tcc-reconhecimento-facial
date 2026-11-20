"""Extração de embeddings faciais com AuraFace (Apache-2.0).

Detector: YuNet (OpenCV Zoo, MIT)
Embedder: AuraFace-v1 (fal.ai, Apache-2.0)

Interface compatível com ArcFaceExtractor:
- Entrada: imagem BGR (np.ndarray)
- Saída: list[FaceDetection], cada um com embedding de 512 dimensões normalizado L2

⚠️ Não é thread-safe: cv2.FaceDetectorYN mantém estado interno (setInputSize).
   Use uma instância por thread ou envolva extract() em lock.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.preprocessing import validate_image_array

logger = logging.getLogger(__name__)

INPUT_SIZE = 112
EMBEDDING_SIZE = 512

# (0,0)                                    (112,0)
#   ┌───────────────────────────────────────┐
#   │                                       │
#   │                                       │
#   │        ●38,51          ●73,51         │  ← olhos (y ≈ 46%)
#   │       (olho E)         (olho D)       │
#   │                                       │
#   │                  ●56,71               │  ← nariz (y ≈ 64%)
#   │                  (nariz)              │
#   │                                       │
#   │     ●41,92              ●70,92        │  ← cantos da boca (y ≈ 82%)
#   │    (boca E)            (boca D)       │
#   │                                       │
#   └───────────────────────────────────────┘
# (0,112)                                 (112,112)
# Template ArcFace de 5 pontos.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceDetection:
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


def _align_face(image: np.ndarray, kps: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Alinha o rosto aos 5 pontos canônicos do ArcFace e recorta 112x112."""
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_DST.copy()
    tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    if tform is None:
        # Fallback: recorta a bbox e redimensiona (NÃO a imagem inteira).
        x, y, w, h = bbox.astype(int)
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box inválido para alinhamento.")
        crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    return cv2.warpAffine(
        image,
        tform,
        (INPUT_SIZE, INPUT_SIZE),
        borderValue=0.0,
        flags=cv2.INTER_LINEAR,
    )


class AuraFaceExtractor:
    """Detecção (YuNet) + alinhamento + embedding (AuraFace ONNX).

    Not thread-safe: use uma instância por thread ou envolva extract() em lock.
    """

    def __init__(
        self,
        auraface_model_path: str | Path | None = None,
        yunet_model_path: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.9,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        # --- AuraFace ONNX ---
        if auraface_model_path is None:
            auraface_model_path = (
                Path(__file__).resolve().parent.parent
                / "Models"
                / "glintr100.onnx"
            )
        auraface_model_path = Path(auraface_model_path)
        if not auraface_model_path.exists():
            raise FileNotFoundError(
                f"Modelo AuraFace não encontrado em '{auraface_model_path}'."
            )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if intra_op_threads is not None:
            so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1

        if providers is None:
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(auraface_model_path), sess_options=so, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info("AuraFace inicializado com providers: %s", self._session.get_providers())

        # --- YuNet Detector ---
        if yunet_model_path is None:
            yunet_model_path = (
                Path(__file__).resolve().parent.parent
                / "Models"
                / "face_detection_yunet_2023mar.onnx"
            )
        yunet_model_path = Path(yunet_model_path)
        if not yunet_model_path.exists():
            raise FileNotFoundError(
                f"Modelo YuNet não encontrado em '{yunet_model_path}'."
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_model_path), "", det_size, score_threshold=score_threshold
        )
        self._det_size = det_size
        self._lock = threading.Lock()

    def extract(self, image: np.ndarray) -> list[FaceDetection]:
        validated = validate_image_array(image)
        if validated.ndim == 2:
            validated = cv2.cvtColor(validated, cv2.COLOR_GRAY2BGR)

        h, w = validated.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(validated)

        detections: list[FaceDetection] = []
        if faces is None:
            return detections

        for face in faces:
            # YuNet face: [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]
            bbox = face[:4].astype(np.float32)
            score = float(face[-1])
            kps = face[4:14].reshape(5, 2).astype(np.float32)

            try:
                aligned = _align_face(validated, kps, bbox)
            except ValueError as exc:
                logger.warning("Alinhamento falhou (bbox=%s): %s", bbox.tolist(), exc)
                continue

            embedding = self._run_onnx(aligned)
            detections.append(
                FaceDetection(embedding=embedding, bbox=bbox, det_score=score)
            )

        return detections

    def _run_onnx(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Pré-processamento AuraFace: RGB, (pixel-127.5)/127.5, NCHW."""
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        embedding = outputs[0].reshape(-1)

        if embedding.size != EMBEDDING_SIZE:
            raise ValueError(
                f"Saída do modelo tem {embedding.size} dimensões, esperado {EMBEDDING_SIZE}."
            )

        return l2_normalize(embedding)
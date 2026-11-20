"""Métricas de desempenho e fairness para reconhecimento facial.

Baseado em ISO/IEC 19795-6:2012 e ISO/IEC 19795-10:2024.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Tuple, Dict


def compute_far_frr(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Calcula FAR e FRR para um dado threshold."""
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores < threshold))
    return far, frr


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
    """Calcula Equal Error Rate (EER) e o threshold correspondente."""
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    with np.errstate(divide='ignore', invalid='ignore'):
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[eer_idx])
    threshold_at_eer = float(thresholds[eer_idx])
    return eer, threshold_at_eer


def compute_roc(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calcula curva ROC e AUC."""
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def bootstrap_ci(data: np.ndarray, statistic_func, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Calcula intervalo de confiança via bootstrap."""
    n = len(data)
    stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(statistic_func(sample))
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def fairness_disparity(metric_by_group: Dict[str, float]) -> Dict[str, float]:
    """Calcula disparidade entre grupos para uma métrica (ex: FAR, FRR, EER).

    Retorna diferença máxima (max-min) e razão (max/min).
    """
    values = np.array(list(metric_by_group.values()))
    names = list(metric_by_group.keys())
    max_val = np.max(values)
    min_val = np.min(values)
    max_group = names[np.argmax(values)]
    min_group = names[np.argmin(values)]
    return {
        'max': float(max_val),
        'min': float(min_val),
        'max_group': max_group,
        'min_group': min_group,
        'difference': float(max_val - min_val),
        'ratio': float(max_val / min_val) if min_val > 0 else float('inf'),
    }
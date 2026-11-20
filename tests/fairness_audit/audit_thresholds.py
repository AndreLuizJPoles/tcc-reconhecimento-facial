"""Teste 3: Thresholds necessários por grupo para um FAR alvo."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models_auraface import AuraFaceExtractor
from src.evaluation import FaceComparator
from src.preprocessing import load_image
from fairness_audit.data_loader import DemographicDataset


def extract_embeddings(extractor, image_paths):
    embeddings = {}
    for path in image_paths:
        img = load_image(path)
        dets = extractor.extract(img)
        if dets:
            embeddings[path] = dets[0].embedding
    return embeddings


def find_threshold_for_far(impostor_scores: np.ndarray, target_far: float) -> float:
    """Menor threshold que garante FAR <= target_far."""
    sorted_scores = np.sort(impostor_scores)[::-1]
    n = len(sorted_scores)
    k = int(np.floor(target_far * n))
    if k >= n:
        return float(sorted_scores[-1]) - 1e-6
    if k == 0:
        return float(sorted_scores[0]) + 1e-6
    return float(sorted_scores[k - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--target-far", type=float, default=1e-4)
    parser.add_argument("--output", default="fairness_audit/threshold_report")
    args = parser.parse_args()

    dataset = DemographicDataset(args.csv, args.base_dir)
    # Mais impostores para estimar FAR com precisão
    pairs_df = dataset.generate_pairs(genuine_per_identity=2, impostor_per_identity=5)

    all_paths = pd.concat([pairs_df['path1'], pairs_df['path2']]).unique().tolist()
    extractor = AuraFaceExtractor()
    embeddings = extract_embeddings(extractor, all_paths)

    valid = [row for _, row in pairs_df.iterrows() if row['path1'] in embeddings and row['path2'] in embeddings]
    pairs_df = pd.DataFrame(valid)
    if pairs_df.empty:
        print("Sem pares válidos.")
        return

    comparator = FaceComparator()
    pairs_df['similarity'] = [comparator.cosine_similarity(embeddings[r['path1']], embeddings[r['path2']]) for _, r in pairs_df.iterrows()]

    thresholds = {}
    for group in pairs_df['group1'].unique():
        impostor = pairs_df[(pairs_df['group1'] == group) & (pairs_df['label'] == 0)]['similarity'].values
        if len(impostor) == 0:
            continue
        thresholds[group] = find_threshold_for_far(impostor, args.target_far)

    if not thresholds:
        print("Sem dados de impostores.")
        return

    print(f"Thresholds para FAR <= {args.target_far}:")
    for g, t in thresholds.items():
        print(f"  {g}: {t:.4f}")

    pd.DataFrame(list(thresholds.items()), columns=['group', 'threshold']).to_csv(
        f"{args.output}_thresholds.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    plt.bar(thresholds.keys(), thresholds.values())
    plt.axhline(y=np.mean(list(thresholds.values())), color='r', linestyle='--', label='Média')
    plt.title(f"Threshold para FAR <= {args.target_far}")
    plt.ylabel("Threshold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output}_barplot.png")
    plt.close()
    print(f"Gráfico salvo em {args.output}_barplot.png")


if __name__ == "__main__":
    main()
"""Teste 2: Auditoria de fairness por grupo demográfico (ISO/IEC 19795-10:2024)."""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models_auraface import AuraFaceExtractor
from src.evaluation import FaceComparator
from src.preprocessing import load_image
from fairness_audit.metrics import compute_far_frr, compute_eer, fairness_disparity
from fairness_audit.data_loader import DemographicDataset


def extract_embeddings(extractor, image_paths):
    embeddings = {}
    for path in image_paths:
        img = load_image(path)
        detections = extractor.extract(img)
        if detections:
            embeddings[path] = detections[0].embedding
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--output", default="fairness_audit/demographic_report")
    parser.add_argument("--threshold", type=float, default=None, help="Se omitido, usa EER global.")
    args = parser.parse_args()

    dataset = DemographicDataset(args.csv, args.base_dir)
    pairs_df = dataset.generate_pairs(genuine_per_identity=2, impostor_per_identity=2)

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

    genuine_all = pairs_df[pairs_df['label'] == 1]['similarity'].values
    impostor_all = pairs_df[pairs_df['label'] == 0]['similarity'].values
    eer_global, eer_thr_global = compute_eer(genuine_all, impostor_all)
    threshold = args.threshold if args.threshold is not None else eer_thr_global
    print(f"Threshold: {threshold:.4f} (EER global: {eer_global:.4f})")

    metrics = {}
    for group in pairs_df['group1'].unique():
        subset = pairs_df[pairs_df['group1'] == group]
        g = subset[subset['label'] == 1]['similarity'].values
        i = subset[subset['label'] == 0]['similarity'].values
        if len(g) == 0 or len(i) == 0:
            continue
        far, frr = compute_far_frr(g, i, threshold)
        eer_g, _ = compute_eer(g, i)
        metrics[group] = {'FAR': far, 'FRR': frr, 'EER': eer_g, 'n_genuine': len(g), 'n_impostor': len(i)}

    if not metrics:
        print("Sem métricas válidas.")
        return

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    print(df_metrics)
    df_metrics.to_csv(f"{args.output}_metrics.csv")

    far_disp = fairness_disparity({g: m['FAR'] for g, m in metrics.items()})
    frr_disp = fairness_disparity({g: m['FRR'] for g, m in metrics.items()})
    eer_disp = fairness_disparity({g: m['EER'] for g, m in metrics.items()})
    print("\nDisparidade FAR:", far_disp)
    print("Disparidade FRR:", frr_disp)
    print("Disparidade EER:", eer_disp)

    with open(f"{args.output}_disparities.txt", 'w') as f:
        f.write(f"FAR disparity: {far_disp}\nFRR disparity: {frr_disp}\nEER disparity: {eer_disp}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ['FAR', 'FRR', 'EER']):
        sns.barplot(x=list(metrics.keys()), y=[m[metric] for m in metrics.values()], ax=ax)
        ax.set_title(f"{metric} por grupo")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args.output}_barplots.png")
    plt.close()
    print(f"Gráficos salvos em {args.output}_barplots.png")


if __name__ == "__main__":
    main()
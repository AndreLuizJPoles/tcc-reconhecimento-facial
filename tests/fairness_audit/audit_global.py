"""Teste 1: Auditoria de desempenho global (ISO/IEC 19795-6:2012)."""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models_auraface import AuraFaceExtractor
from src.evaluation import FaceComparator
from src.preprocessing import load_image
from fairness_audit.metrics import compute_eer, compute_roc, compute_far_frr
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
    parser = argparse.ArgumentParser(description="Auditoria global de desempenho.")
    parser.add_argument("--csv", required=True, help="CSV com image_path, identity, demographic_group")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--output", default="fairness_audit/global_report")
    args = parser.parse_args()

    dataset = DemographicDataset(args.csv, args.base_dir)
    pairs_df = dataset.generate_pairs(genuine_per_identity=2, impostor_per_identity=2)

    all_paths = pd.concat([pairs_df['path1'], pairs_df['path2']]).unique().tolist()
    extractor = AuraFaceExtractor()
    embeddings = extract_embeddings(extractor, all_paths)

    valid = [row for _, row in pairs_df.iterrows() if row['path1'] in embeddings and row['path2'] in embeddings]
    pairs_df = pd.DataFrame(valid)
    if pairs_df.empty:
        print("Nenhum par válido.")
        return

    comparator = FaceComparator()
    sims = [comparator.cosine_similarity(embeddings[r['path1']], embeddings[r['path2']]) for _, r in pairs_df.iterrows()]
    pairs_df['similarity'] = sims

    genuine = pairs_df[pairs_df['label'] == 1]['similarity'].values
    impostor = pairs_df[pairs_df['label'] == 0]['similarity'].values
    eer, eer_thr = compute_eer(genuine, impostor)
    fpr, tpr, _, auc_val = compute_roc(genuine, impostor)
    far, frr = compute_far_frr(genuine, impostor, eer_thr)

    print(f"EER: {eer:.4f} (threshold: {eer_thr:.4f})")
    print(f"FAR @ EER: {far:.4f} | FRR @ EER: {frr:.4f} | AUC: {auc_val:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FAR")
    plt.ylabel("1 - FRR")
    plt.title("Curva ROC - Global")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output}_roc.png")
    plt.close()

    pd.DataFrame([{'eer': eer, 'eer_threshold': eer_thr, 'far_at_eer': far, 'frr_at_eer': frr, 'auc': auc_val}]).to_csv(
        f"{args.output}_metrics.csv", index=False
    )
    print(f"Relatório salvo em {args.output}_*")


if __name__ == "__main__":
    main()
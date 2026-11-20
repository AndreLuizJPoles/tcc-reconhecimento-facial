"""Teste 4: Identificação 1:N por grupo demográfico."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models_auraface import AuraFaceExtractor
from src.evaluation import FaceComparator
from src.gallery import FaceGallery, GalleryRecord
from src.preprocessing import load_image


def build_gallery_from_csv(csv_path, base_dir, extractor):
    df = pd.read_csv(csv_path)
    required = {'image_path', 'identity', 'demographic_group'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV deve conter {required}")
    base_dir = Path(base_dir) if base_dir else Path('.')

    records = []
    for _, row in df.iterrows():
        path = Path(row['image_path'])
        if not path.is_absolute():
            path = base_dir / path
        img = load_image(path)
        dets = extractor.extract(img)
        if not dets:
            continue
        d = dets[0]
        records.append(GalleryRecord(
            identity=row['identity'],
            image_path=str(path),
            embedding=d.embedding,
            bbox=d.bbox,
            det_score=d.det_score,
        ))
    return FaceGallery(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-csv", required=True)
    parser.add_argument("--query-csv", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--output", default="fairness_audit/1toN_report")
    args = parser.parse_args()

    extractor = AuraFaceExtractor()
    gallery = build_gallery_from_csv(args.gallery_csv, args.base_dir, extractor)
    if not gallery.records:
        print("Galeria vazia.")
        return

    comparator = FaceComparator(threshold=args.threshold)
    df_query = pd.read_csv(args.query_csv)
    base_dir = Path(args.base_dir) if args.base_dir else Path('.')

    results = []
    for _, row in df_query.iterrows():
        path = Path(row['image_path'])
        if not path.is_absolute():
            path = base_dir / path
        img = load_image(path)
        dets = extractor.extract(img)
        if not dets:
            continue
        query_emb = dets[0].embedding
        ident = comparator.identify(query_emb, gallery.records, top_k=1)
        predicted = ident.identity if ident.matched else None
        results.append({
            'group': row['demographic_group'],
            'true_identity': row['identity'],
            'predicted': predicted,
            'correct': predicted == row['identity'],
            'false_accept': (predicted is not None and predicted != row['identity']),
        })

    if not results:
        print("Sem queries válidas.")
        return

    df_res = pd.DataFrame(results)
    metrics = []
    for group, subset in df_res.groupby('group'):
        n = len(subset)
        metrics.append({
            'group': group,
            'n': n,
            'rank1_accuracy': subset['correct'].mean(),
            'false_accept_rate': subset['false_accept'].mean(),
        })

    df_metrics = pd.DataFrame(metrics)
    print(df_metrics)
    df_metrics.to_csv(f"{args.output}_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(df_metrics['group'], df_metrics['rank1_accuracy'])
    axes[0].set_title('Rank-1 Accuracy por grupo')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(df_metrics['group'], df_metrics['false_accept_rate'])
    axes[1].set_title('False Accept Rate por grupo')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{args.output}_barplots.png")
    plt.close()
    print(f"Gráficos salvos em {args.output}_barplots.png")


if __name__ == "__main__":
    main()
"""Carregamento de dados para auditoria de fairness."""

import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
from itertools import combinations


class DemographicDataset:
    """Dataset de imagens faciais com rótulos demográficos.

    Espera um CSV com colunas: image_path, identity, demographic_group.
    """

    def __init__(self, csv_path: str | Path, base_dir: str | Path | None = None):
        self.df = pd.read_csv(csv_path)
        required = {'image_path', 'identity', 'demographic_group'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV deve conter colunas: {required}")
        self.base_dir = Path(base_dir) if base_dir else Path('.')

    def get_image_path(self, row) -> Path:
        p = Path(row['image_path'])
        return p if p.is_absolute() else self.base_dir / p

    def generate_pairs(self, genuine_per_identity: int = 1, impostor_per_identity: int = 1) -> pd.DataFrame:
        """Gera pares genuínos e impostores a partir do dataset.

        Retorna DataFrame com colunas: path1, path2, label (1/0), group1, group2.
        """
        pairs = []
        grouped = self.df.groupby('identity')
        identities = list(grouped.groups.keys())

        # Pares genuínos
        for identity, group in grouped:
            if len(group) < 2:
                continue
            indices = group.index.tolist()
            count = 0
            for i, j in combinations(indices, 2):
                row_i = self.df.loc[i]
                row_j = self.df.loc[j]
                pairs.append({
                    'path1': str(self.get_image_path(row_i)),
                    'path2': str(self.get_image_path(row_j)),
                    'label': 1,
                    'group1': row_i['demographic_group'],
                    'group2': row_j['demographic_group'],
                })
                count += 1
                if count >= genuine_per_identity:
                    break

        # Pares impostores (amostragem aleatória)
        rng = np.random.default_rng(42)
        impostor_pairs = set()
        target_impostors = impostor_per_identity * len(identities)
        while len(impostor_pairs) < target_impostors and len(impostor_pairs) < len(self.df) * (len(self.df) - 1) / 2:
            i, j = rng.choice(len(self.df), size=2, replace=False)
            row_i = self.df.iloc[i]
            row_j = self.df.iloc[j]
            if row_i['identity'] == row_j['identity']:
                continue
            key = tuple(sorted([str(self.get_image_path(row_i)), str(self.get_image_path(row_j))]))
            if key in impostor_pairs:
                continue
            impostor_pairs.add(key)
            pairs.append({
                'path1': str(self.get_image_path(row_i)),
                'path2': str(self.get_image_path(row_j)),
                'label': 0,
                'group1': row_i['demographic_group'],
                'group2': row_j['demographic_group'],
            })

        return pd.DataFrame(pairs)
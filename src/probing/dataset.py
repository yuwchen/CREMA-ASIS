"""Dataset classes for cached embedding-based probing.

Consolidates ``CachedEmbeddingDataset`` and ``collate_fn`` that were
duplicated identically in the four extraction/probing scripts.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    """Dataset that serves pre-computed pooled embeddings for a given layer.

    Args:
        embeddings: ``{file_path: {layer_key: {"mean": tensor, ...}}}``.
        labels: Integer label for each file (same order as *files*).
        layer_key: Which layer to read from (int index or string name).
        pooling: Pooling strategy key (``"mean"``, ``"last"``, etc.).
        files: Ordered list of file paths.  Embeddings are re-ordered to
            match this list so that label alignment is guaranteed.
    """

    def __init__(
        self,
        embeddings: Dict,
        labels: List[int],
        layer_key,
        pooling: str,
        files: Optional[List[str]] = None,
    ):
        if files is not None:
            embeddings = self._align(embeddings, files)

        self.file_paths = list(embeddings.keys())
        self.embeddings = embeddings
        self.labels = labels
        self.layer_key = layer_key
        self.pooling = pooling

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        feat = self.embeddings[fp][self.layer_key][self.pooling]

        # Normalise to float32 tensor regardless of storage format
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat.astype(np.float32))
        elif feat.dtype != torch.float32:
            feat = feat.float()

        return feat, self.labels[idx]

    # ------------------------------------------------------------------

    @staticmethod
    def _align(embeddings: Dict, file_list: List[str]) -> OrderedDict:
        """Re-order *embeddings* to match *file_list* ordering."""
        aligned = OrderedDict()
        missing = []
        for fp in file_list:
            if fp in embeddings:
                aligned[fp] = embeddings[fp]
            else:
                missing.append(fp)
        if missing:
            print(f"WARNING: {len(missing)} files missing from embeddings "
                  f"(first 5: {missing[:5]})")
        extra = set(embeddings.keys()) - set(file_list)
        if extra:
            print(f"WARNING: {len(extra)} extra files in embeddings not in file_list")
        return aligned


def collate_fn(batch):
    """Simple collate for ``(feature_tensor, int_label)`` pairs."""
    features, labels = zip(*batch)
    return torch.stack(features), torch.tensor(labels)

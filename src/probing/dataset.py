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
            embeddings, labels = self._align(embeddings, files, labels)

        self.file_paths = list(embeddings.keys())
        self.embeddings = embeddings
        self.labels = labels
        self.layer_key = layer_key
        self.pooling = pooling

        if len(self.labels) != len(self.file_paths):
            raise ValueError(
                f"Label/feature mismatch: {len(self.labels)} labels for "
                f"{len(self.file_paths)} embeddings."
            )

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
    def _align(
        embeddings: Dict, file_list: List[str], labels: List[int]
    ) -> tuple[OrderedDict, List[int]]:
        """Re-order *embeddings* to match *file_list*, dropping missing files.

        Labels are filtered alongside the files.  Dropping a file without
        dropping its label would shift every later label by one and silently
        corrupt the probe, so the two are always kept in step.
        """
        if len(labels) != len(file_list):
            raise ValueError(
                f"Got {len(labels)} labels for {len(file_list)} files; "
                f"they must be the same length and in the same order."
            )

        aligned = OrderedDict()
        kept_labels: List[int] = []
        missing = []
        for fp, label in zip(file_list, labels):
            if fp in embeddings:
                aligned[fp] = embeddings[fp]
                kept_labels.append(label)
            else:
                missing.append(fp)

        if missing:
            print(f"WARNING: {len(missing)} files missing from embeddings, "
                  f"dropped with their labels (first 5: {missing[:5]})")
        extra = set(embeddings.keys()) - set(file_list)
        if extra:
            print(f"WARNING: {len(extra)} extra files in embeddings not in file_list")
        return aligned, kept_labels


def collate_fn(batch):
    """Simple collate for ``(feature_tensor, int_label)`` pairs."""
    features, labels = zip(*batch)
    return torch.stack(features), torch.tensor(labels)

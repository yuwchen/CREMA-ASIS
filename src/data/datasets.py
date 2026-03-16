"""Unified dataset loading with per-dataset column mappings.

Consolidates ``dataloader()``, ``dataloader_base()``, and ``dataloader_meld()``
from the original per-model scripts into a single interface that normalises
every dataset into a common schema:

    filepath | acoustic | semantic | split

Each dataset's idiosyncratic column names are declared in a :class:`DatasetConfig`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Declares how a CSV maps to the unified schema.

    Attributes:
        csv_path: Path to the CSV file.
        split_column: Column that holds ``train`` / ``val`` / ``test`` labels.
        acoustic_column: Column with acoustic emotion labels.
        semantic_column: Column with semantic sentiment labels.
            If ``None``, all samples default to ``"neutral"``.
        filepath_column: Column with pre-built file paths (mutually exclusive
            with ``wav_dir`` + ``wav_column``).
        wav_dir: Directory prefix to join with ``wav_column``.
        wav_column: Column containing just the filename (joined with ``wav_dir``).
        repeat: How many times to repeat this dataset when merging.  Useful to
            up-sample a smaller base dataset.
    """

    csv_path: str
    split_column: str = "split"
    acoustic_column: str = "acoustic"
    semantic_column: Optional[str] = "semantic"
    filepath_column: Optional[str] = "filepath"
    split: str = None
    wav_dir: Optional[str] = None
    wav_column: Optional[str] = None
    repeat: int = 1


# ---------------------------------------------------------------------------
# Pre-defined configs for known datasets
# ---------------------------------------------------------------------------

CREMAD_ANNOTATED = DatasetConfig(
    csv_path="data/cremad_all_clean_w_sad_filtered.csv",
    acoustic_column="acoustic",
    semantic_column="semantic",
    filepath_column=None,
    wav_dir="data/cremad-sync/cremad-sync-wsad/",
    wav_column="output_name",
)

CREMAD_BASE = DatasetConfig(
    csv_path="data/crema-d_en_split.csv",
    acoustic_column="emotion",
    semantic_column=None,  # defaults to "neutral"
    filepath_column=None,
    wav_dir="data/CREMA-D/AudioWAV_en/",
    wav_column="wavname",
)

MELD_TRAIN = DatasetConfig(
    csv_path="data/MELD.Raw/meld_train.csv",
    acoustic_column="Emotion",
    semantic_column="Sentiment",
    filepath_column=None,
    wav_dir="data/MELD.Raw/train_wav",
    wav_column="wavname",
)

MELD_VAL = DatasetConfig(
    csv_path="data/MELD.Raw/meld_val.csv",
    acoustic_column="Emotion",
    semantic_column="Sentiment",
    filepath_column=None,
    wav_dir="data/MELD.Raw/dev_wav",
    wav_column="wavname",
)

"""
# Ran locally
CREMAD_ANNOTATED = DatasetConfig(
    csv_path="../emotional_tts/cremad_all_clean_w_sad_filtered.csv",
    acoustic_column="acoustic",
    semantic_column="semantic",
    filepath_column=None,
    wav_dir="../emotional_tts/cremad-sync/cremad-sync-wsad/",
    wav_column="output_name",
)

CREMAD_BASE = DatasetConfig(
    csv_path="../emotional_tts/crema-d_en_split.csv",
    acoustic_column="emotion",
    semantic_column=None,  # defaults to "neutral"
    filepath_column=None,
    wav_dir="../emotional_tts/AudioWAV_en/",
    wav_column="wavname",
)

MELD_TRAIN = DatasetConfig(
    csv_path="../emotional_tts/MELD/MELD.Raw/train_sent_emo_with_paths.csv",
    acoustic_column="Emotion",
    semantic_column="Sentiment",
    split='train',
    filepath_column=None,
    wav_dir="../emotional_tts/MELD/MELD.Raw",
    wav_column="Audio_Path",
)

MELD_VAL = DatasetConfig(
    csv_path="../emotional_tts/MELD/MELD.Raw/val_sent_emo_with_paths.csv",
    acoustic_column="Emotion",
    semantic_column="Sentiment",
    split='val',
    filepath_column=None,
    wav_dir="../emotional_tts/MELD/MELD.Raw",
    wav_column="Audio_Path",
)
"""

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load a single dataset CSV and normalise it to the unified schema.

    The returned DataFrame always has columns:
    ``filepath``, ``acoustic``, ``semantic``, ``split``.

    Args:
        config: A :class:`DatasetConfig` describing column mappings.

    Returns:
        Normalised :class:`~pandas.DataFrame`.
    """
    df = pd.read_csv(config.csv_path)

    # -- Normalise column names --
    rename_map = {}
    if config.acoustic_column != "acoustic":
        rename_map[config.acoustic_column] = "acoustic"
    if config.semantic_column and config.semantic_column != "semantic":
        rename_map[config.semantic_column] = "semantic"
    if config.split_column != "split":
        rename_map[config.split_column] = "split"
    if rename_map:
        df = df.rename(columns=rename_map)

    # -- Build filepath if not directly available --
    if config.filepath_column and config.filepath_column in df.columns:
        df = df.rename(columns={config.filepath_column: "filepath"})
    elif config.wav_dir and config.wav_column:
        df["filepath"] = df[config.wav_column].apply(
            lambda x: os.path.join(config.wav_dir, x)
        )
    # else: assume "filepath" already exists

    # -- Default semantic to "neutral" when absent --
    if config.semantic_column is None or "semantic" not in df.columns:
        df["semantic"] = "neutral"

    # Keep only the columns we need (plus any extras the caller may want)
    required = {"filepath", "acoustic", "semantic", "split"}
    for col in required:
        if col not in df.columns:
            if config.split:
                df['split'] = config.split
            else:
                raise ValueError(
                    f"Column '{col}' missing after normalisation of {config.csv_path}. "
                    f"Available: {list(df.columns)}"
                )

    return df


def load_and_merge_datasets(
    configs: List[DatasetConfig],
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Load, merge, and split multiple datasets for fine-tuning.

    Args:
        configs: List of :class:`DatasetConfig` instances.
        shuffle: Whether to shuffle after merging.
        seed: Random seed for shuffling.

    Returns:
        Tuple of ``(train_samples, val_samples)`` where each sample is a dict
        with keys ``audio_path`` and ``target``.
    """
    frames = []
    for cfg in configs:
        df = load_dataset(cfg)
        if cfg.repeat > 1:
            df = pd.concat([df] * cfg.repeat, ignore_index=True)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_template = (
        '{{\n    "acoustic_emotion": "{acoustic}",\n'
        '    "semantic_sentiment": "{semantic}"\n    }}'
    )

    def _to_samples(sub_df: pd.DataFrame) -> List[dict]:
        samples = []
        for _, row in sub_df.iterrows():
            target = target_template.format(
                acoustic=row["acoustic"], semantic=row["semantic"]
            )
            samples.append({"audio_path": row["filepath"], "target": target})
        return samples

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_samples = _to_samples(train_df)
    val_samples = _to_samples(val_df)

    if shuffle:
        random.seed(seed)
        random.shuffle(train_samples)
        random.shuffle(val_samples)

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Data loading for probing / feature extraction (CSV-based splits)
# ---------------------------------------------------------------------------

def load_data_for_probing(
    csv_path: str,
    data_dir: str,
    sample_percentage: int = 100,
    wer_threshold: float = 0.5,
) -> dict:
    """Load data splits from a CSV for embedding extraction and probing.

    This is the shared implementation of the ``load_data_from_csv`` function
    that was duplicated across the three feature-extraction scripts.

    Args:
        csv_path: Path to CSV with columns ``output_name, acoustic, semantic, split``
            and optionally ``wer``.
        data_dir: Directory containing the audio files.
        sample_percentage: Percentage of data to use (1–100).
        wer_threshold: Maximum WER for test-set inclusion.

    Returns:
        Dict with keys:
        ``train_files``, ``train_acoustic``, ``train_semantic``,
        ``val_files``, ``val_acoustic``, ``val_semantic``,
        ``test_files``, ``test_acoustic``, ``test_semantic``,
        ``acoustic_to_idx``, ``semantic_to_idx``.
    """
    print(f"Loading data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    # Construct full paths
    df["full_path"] = df["output_name"].apply(lambda x: os.path.join(data_dir, x))

    # Verify files exist
    missing = df[~df["full_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} files not found!")
        print(f"First few missing: {missing['full_path'].tolist()[:5]}")
        df = df[df["full_path"].apply(os.path.exists)]
        print(f"Continuing with {len(df)} files that exist")

    # Optional sub-sampling
    if sample_percentage < 100:
        np.random.seed(42)
        df = df.sample(frac=sample_percentage / 100, random_state=42)
        print(f"Sampled {len(df)} files ({sample_percentage}%)")

    # Split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if "wer" in df.columns:
        test_df = test_df[test_df["wer"] < wer_threshold]

    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def _extract(sub_df):
        return (
            sub_df["full_path"].tolist(),
            sub_df["acoustic"].tolist(),
            sub_df["semantic"].tolist(),
        )

    train_files, train_acoustic, train_semantic = _extract(train_df)
    val_files, val_acoustic, val_semantic = _extract(val_df)
    test_files, test_acoustic, test_semantic = _extract(test_df)

    # Label mappings
    all_acoustic = train_acoustic + val_acoustic + test_acoustic
    all_semantic = train_semantic + val_semantic + test_semantic
    acoustic_to_idx = {l: i for i, l in enumerate(sorted(set(all_acoustic)))}
    semantic_to_idx = {l: i for i, l in enumerate(sorted(set(all_semantic)))}

    print(f"Acoustic emotions: {list(acoustic_to_idx.keys())}")
    print(f"Semantic labels: {list(semantic_to_idx.keys())}")

    return {
        "train_files": train_files,
        "train_acoustic": train_acoustic,
        "train_semantic": train_semantic,
        "val_files": val_files,
        "val_acoustic": val_acoustic,
        "val_semantic": val_semantic,
        "test_files": test_files,
        "test_acoustic": test_acoustic,
        "test_semantic": test_semantic,
        "acoustic_to_idx": acoustic_to_idx,
        "semantic_to_idx": semantic_to_idx,
    }

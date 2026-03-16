"""Embedding cache management.

Provides incremental extraction with resume capability, memory-efficient
disk I/O, and atomic saves.  This module consolidates the caching logic
that was duplicated across the four extraction scripts.
"""

from __future__ import annotations

import gc
import hashlib
import os
import pickle
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cache path generation
# ---------------------------------------------------------------------------

def get_legacy_cache_path(
    model_name: str,
    data_identifier: str,
    prompt_type: str,
    selected_layers: list,
    use_float16: bool = True,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    cache_dir: str = "qwen2_layer_embedding_cache",
) -> str:
    """Generate a cache path compatible with the old extraction format (v3).

    Use this to load caches produced by the legacy ``get_cache_path`` that
    used global ``SELECTED_LAYERS`` / ``USE_FLOAT16`` constants, a ``_v3``
    key suffix, and ``_selective.pkl`` filenames.

    Args:
        model_name: Model name as it appeared in the old cache key.
        data_identifier: CSV basename without extension (no split suffix).
        prompt_type: ``"audio_only"`` or ``"audio_text"``.
        selected_layers: Layer indices used during extraction.
        use_float16: Whether embeddings were stored in FP16.
        use_lora: Whether LoRA was applied.
        lora_path: Path to LoRA checkpoint (used in hash if use_lora=True).
        cache_dir: Legacy cache directory (default ``qwen2_layer_embedding_cache``).

    Returns:
        Path to the legacy ``.pkl`` cache file.
    """
    layers_str = "_".join(map(str, selected_layers))
    dtype_str = "fp16" if use_float16 else "fp32"
    model_shortname = model_name.split("/")[-1] if "/" in model_name else model_name

    if use_lora and lora_path:
        lora_name = os.path.basename(lora_path)
        cache_key = f"{model_shortname}_{data_identifier}_{prompt_type}_{dtype_str}_{layers_str}_lora_{lora_name}_v3"
    else:
        cache_key = f"{model_shortname}_{data_identifier}_{prompt_type}_{dtype_str}_{layers_str}_base_v3"

    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    os.makedirs(cache_dir, exist_ok=True)

    lora_tag = "lora_" if use_lora else ""
    filename = f"embeddings_{lora_tag}{cache_hash}_{prompt_type}_{dtype_str}_selective.pkl"
    print(f"Legacy cache file: {filename}")
    return os.path.join(cache_dir, filename)


def get_cache_path(
    model_name: str,
    data_identifier: str,
    split: str,
    prompt_type: str,
    selected_layers: list,
    use_float16: bool = True,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    cache_dir: str = "embedding_cache",
) -> str:
    """Generate a deterministic cache file path.

    The path encodes model name, data split, prompt type, layer selection,
    dtype, and LoRA checkpoint so that different configurations never
    collide.

    Note: ``sample_percentage`` is intentionally **not** included so that
    incremental extraction (10 %% → 100 %%) reuses the same cache file.

    Args:
        model_name: HuggingFace model identifier.
        data_identifier: E.g. CSV filename without extension.
        split: ``"train"`` / ``"val"`` / ``"test"``.
        prompt_type: ``"audio_text"`` or ``"audio_only"``.
        selected_layers: List of layer indices.
        use_float16: Whether embeddings are stored in FP16.
        use_lora: Whether LoRA is applied.
        lora_path: Path to LoRA checkpoint (used in hash).
        cache_dir: Parent directory for cache files.

    Returns:
        Absolute path to the ``.pkl`` cache file.
    """
    layers_str = "_".join(map(str, selected_layers))
    dtype_str = "fp16" if use_float16 else "fp32"
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    if use_lora and lora_path:
        lora_name = os.path.basename(lora_path)
        key = f"{model_short}_{data_identifier}_{split}_{prompt_type}_{dtype_str}_{layers_str}_lora_{lora_name}"
    else:
        key = f"{model_short}_{data_identifier}_{split}_{prompt_type}_{dtype_str}_{layers_str}_base"

    cache_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    os.makedirs(cache_dir, exist_ok=True)

    lora_tag = "lora_" if use_lora else ""
    filename = f"embeddings_{lora_tag}{cache_hash}_{prompt_type}_{dtype_str}.pkl"
    return os.path.join(cache_dir, filename)


# ---------------------------------------------------------------------------
# Atomic save / load helpers
# ---------------------------------------------------------------------------

def _atomic_save(data: dict, path: str) -> None:
    """Write *data* to *path* via a temporary file (atomic rename)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.rename(tmp, path)


def load_existing_cache(cache_path: str) -> Tuple[dict, Set[str]]:
    """Load an existing cache, returning ``(embeddings, processed_files)``."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
            return embeddings, set(embeddings.keys())
        except Exception as e:
            print(f"Error loading cache: {e} — starting fresh")
    return {}, set()


def merge_and_save_cache(
    new_embeddings: dict,
    cache_path: str,
    verify: bool = True,
) -> int:
    """Merge *new_embeddings* into the on-disk cache and save atomically.

    Returns:
        Total number of files in the cache after merge.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            disk = pickle.load(f)
        disk.update(new_embeddings)
    else:
        disk = new_embeddings

    total = len(disk)
    _atomic_save(disk, cache_path)

    if verify:
        with open(cache_path, "rb") as f:
            check = pickle.load(f)
        assert len(check) == total, "Cache size mismatch after save!"
        del check

    return total


# ---------------------------------------------------------------------------
# Main extraction loop with caching
# ---------------------------------------------------------------------------

def extract_and_cache_all(
    extract_fn: Callable[[str], dict],
    file_list: List[str],
    cache_path: str,
    save_every_n: int = 500,
    memory_efficient: bool = True,
) -> dict:
    """Extract embeddings for *file_list* with incremental caching.

    Already-processed files are skipped.  In memory-efficient mode the
    in-RAM accumulator is flushed to disk every *save_every_n* files.

    Args:
        extract_fn: Callable that takes an audio path and returns a dict of
            ``{layer_key: {"mean": tensor, "last": tensor, ...}}``.
        file_list: Audio file paths to process.
        cache_path: Where to read/write the ``.pkl`` cache.
        save_every_n: Flush interval (memory-efficient mode).
        memory_efficient: If ``True``, RAM is cleared after each flush.

    Returns:
        Complete embeddings dict.
    """
    # -- Determine already-processed files --
    if os.path.exists(cache_path):
        cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
        if memory_efficient:
            with open(cache_path, "rb") as f:
                tmp = pickle.load(f)
            processed = set(tmp.keys())
            del tmp
            gc.collect()
            embeddings: dict = {}
        else:
            embeddings, processed = load_existing_cache(cache_path)
        print(f"Cache: {len(processed)} files ({cache_mb:.1f} MB)")
    else:
        processed = set()
        embeddings = {}

    files_todo = [f for f in file_list if f not in processed]
    if not files_todo:
        print("All files already cached.")
        if memory_efficient:
            embeddings, _ = load_existing_cache(cache_path)
        return embeddings

    print(f"Extracting {len(files_todo)}/{len(file_list)} files "
          f"({'memory-efficient' if memory_efficient else 'fast'} mode)")

    count = 0
    for audio_path in tqdm(files_todo, desc="Extracting"):
        try:
            embeddings[audio_path] = extract_fn(audio_path)
            count += 1

            if count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if count >= save_every_n:
                if memory_efficient:
                    merge_and_save_cache(embeddings, cache_path)
                    embeddings = {}
                else:
                    _atomic_save(embeddings, cache_path)
                count = 0
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on {audio_path}: {e}")
            _atomic_save(embeddings, cache_path)
            raise

    # -- Final save --
    if memory_efficient:
        if embeddings:
            merge_and_save_cache(embeddings, cache_path)
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        _atomic_save(embeddings, cache_path)

    cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Extraction complete: {len(embeddings)} files, {cache_mb:.1f} MB")
    return embeddings

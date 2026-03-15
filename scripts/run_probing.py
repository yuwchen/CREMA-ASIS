#!/usr/bin/env python3
"""Step 6: Run linear probing experiments on cached embeddings.

Usage:
    python scripts/run_probing.py \
        --model qwen2-audio --component llm \
        --model-config configs/models/qwen2_audio.yaml \
        --probe-config configs/probing/default.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/ \
        --cache-dir embedding_cache \
        --results-dir probe_results

    python scripts/run_probing.py \
        --model kimi-audio --component llm \
        --model-config configs/models/kimi_audio.yaml \
        --probe-config configs/probing/default.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/ \
        --cache-dir embedding_cache \
        --results-dir probe_results \
        --lora-path finetuned_models/kimi-lora/checkpoint-7500
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

from src.data.datasets import load_data_for_probing
from src.extraction.cache import get_cache_path, get_legacy_cache_path
from src.probing.linear_probe import run_layer_probe_experiment
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Linear probing on cached embeddings")
    parser.add_argument(
        "--model", required=True,
        choices=["qwen2-audio", "kimi-audio", "audio-flamingo3"],
    )
    parser.add_argument("--component", default="llm", choices=["llm", "whisper"])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--probe-config", required=True)
    parser.add_argument("--csv", required=True, help="Data CSV path")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", default="embedding_cache")
    parser.add_argument("--results-dir", default="probe_results")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--sample-pct", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--not-float16", action="store_true")
    parser.add_argument(
        "--legacy-cache-dir", default=None,
        help="Path to a legacy flat cache directory (no model/component subfolders). "
             "Uses the old v3 key format and _selective.pkl filenames.",
    )
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    probe_cfg = load_yaml(args.probe_config)
    use_float16 = not args.not_float16
    use_lora = args.lora_path is not None

    comp_cfg = model_cfg["extraction"].get(args.component, {})
    selected_layers = comp_cfg.get("selected_layers", [0, 1, 6, 13, 20, 27])

    # -- Load data --
    data = load_data_for_probing(args.csv, args.data_dir, args.sample_pct)
    data_id = os.path.basename(args.csv).replace(".csv", "")

    # -- Load cached embeddings --
    split_embeddings = {}
    for split in ("train", "val", "test"):
        if args.legacy_cache_dir:
            cache_path = get_legacy_cache_path(
                model_name=model_cfg["model_name"],
                data_identifier=f"{data_id}_{split}",
                prompt_type="audio_text",
                selected_layers=selected_layers,
                use_float16=use_float16,
                use_lora=use_lora,
                lora_path=args.lora_path,
                cache_dir=args.legacy_cache_dir,
            )
        else:
            cache_path = get_cache_path(
                model_name=model_cfg["model_name"],
                data_identifier=f"{data_id}_{split}",
                split=split,
                prompt_type="audio_text",
                selected_layers=selected_layers,
                use_float16=use_float16,
                use_lora=use_lora,
                lora_path=args.lora_path,
                cache_dir=os.path.join(args.cache_dir, args.model, args.component),
            )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run scripts/extract_embeddings.py first."
            )
        print(f"Loading {split} embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        if args.legacy_cache_dir:
            # Legacy keys are relative paths; remap to absolute using --data-dir.
            data_dir = args.data_dir.rstrip("/")
            raw = {
                os.path.join(data_dir, os.path.basename(k)): v
                for k, v in raw.items()
            }
        split_embeddings[split] = raw

    # -- Run probing for each seed --
    seeds = probe_cfg.get("seeds", [0, 1, 2, 3, 4])
    pooling = comp_cfg.get("pooling_strategies", probe_cfg["pooling_strategies"])

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'#' * 80}")
        print(f"# SEED {seed}")
        print(f"{'#' * 80}")

        seed_dir = os.path.join(
            args.results_dir, args.model, args.component, f"seed_{seed}"
        )
        os.makedirs(seed_dir, exist_ok=True)

        # -- Acoustic emotion probing --
        print(f"\n{'=' * 60}")
        print("ACOUSTIC EMOTION PROBING")
        print(f"{'=' * 60}")
        run_layer_probe_experiment(
            task_name="Acoustic Emotion",
            train_embeddings=split_embeddings["train"],
            val_embeddings=split_embeddings["val"],
            test_embeddings=split_embeddings["test"],
            train_files=data["train_files"],
            val_files=data["val_files"],
            test_files=data["test_files"],
            train_labels=data["train_acoustic"],
            val_labels=data["val_acoustic"],
            test_labels=data["test_acoustic"],
            label_to_idx=data["acoustic_to_idx"],
            pooling_strategies=pooling,
            learning_rates=probe_cfg["learning_rates"],
            batch_size=probe_cfg["batch_size"],
            num_epochs=probe_cfg["num_epochs"],
            use_scheduler=probe_cfg["use_scheduler"],
            results_dir=seed_dir,
            lora_path=args.lora_path,
            device=args.device,
        )

        # -- Semantic sentiment probing --
        print(f"\n{'=' * 60}")
        print("SEMANTIC LABEL PROBING")
        print(f"{'=' * 60}")
        run_layer_probe_experiment(
            task_name="Semantic Label",
            train_embeddings=split_embeddings["train"],
            val_embeddings=split_embeddings["val"],
            test_embeddings=split_embeddings["test"],
            train_files=data["train_files"],
            val_files=data["val_files"],
            test_files=data["test_files"],
            train_labels=data["train_semantic"],
            val_labels=data["val_semantic"],
            test_labels=data["test_semantic"],
            label_to_idx=data["semantic_to_idx"],
            pooling_strategies=pooling,
            learning_rates=probe_cfg["learning_rates"],
            batch_size=probe_cfg["batch_size"],
            num_epochs=probe_cfg["num_epochs"],
            use_scheduler=probe_cfg["use_scheduler"],
            results_dir=seed_dir,
            lora_path=args.lora_path,
            device=args.device,
        )

    print(f"\nAll probing experiments complete. Results in {args.results_dir}/")


if __name__ == "__main__":
    main()

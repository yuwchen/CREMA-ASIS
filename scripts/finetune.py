#!/usr/bin/env python3
"""Step 2: LoRA fine-tuning for any supported model.

Usage:
    python scripts/finetune.py \
        --model qwen2-audio \
        --model-config configs/models/qwen2_audio.yaml \
        --train-config configs/training/default.yaml \
        --output-dir finetuned_models/qwen2-audio-lora \
        --datasets cremad_annotated cremad_base meld

    python scripts/finetune.py \
        --model kimi-audio \
        --model-config configs/models/kimi_audio.yaml \
        --train-config configs/training/default.yaml \
        --output-dir finetuned_models/kimi-audio-lora \
        --datasets cremad_annotated cremad_base meld
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

from src.data.datasets import (
    CREMAD_ANNOTATED,
    CREMAD_BASE,
    MELD_TRAIN,
    MELD_VAL,
    DatasetConfig,
    load_and_merge_datasets,
)
from src.training.finetune import run_finetuning
from src.utils.seed import set_seed


# Shorthand names → DatasetConfig lists
DATASET_PRESETS = {
    "cremad_annotated": [CREMAD_ANNOTATED],
    "cremad_base": [CREMAD_BASE],
    "meld": [MELD_TRAIN, MELD_VAL],
}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument(
        "--model", required=True,
        choices=["qwen2-audio", "kimi-audio", "flamingo3"],
    )
    parser.add_argument("--model-config", required=True, help="Model YAML config")
    parser.add_argument("--train-config", required=True, help="Training YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--datasets", nargs="+", default=["cremad_annotated", "cremad_base", "meld"],
        help="Dataset presets to combine",
    )
    parser.add_argument("--prompt", default="configs/prompts/emotion_sentiment.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Gather dataset configs
    configs = []
    for name in args.datasets:
        if name in DATASET_PRESETS:
            configs.extend(DATASET_PRESETS[name])
        else:
            raise ValueError(f"Unknown dataset preset: {name}")

    train_samples, val_samples = load_and_merge_datasets(configs, seed=args.seed)
    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    run_finetuning(
        model_type=args.model,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        train_dataset=train_samples,
        val_dataset=val_samples,
        output_dir=args.output_dir,
        prompt_path=args.prompt,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

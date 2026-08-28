#!/usr/bin/env python3
"""Step 3: LoRA fine-tuning for any supported model.

Usage:
    python scripts/finetune.py \
        --model qwen2-audio \
        --model-config configs/models/qwen2_audio.yaml \
        --train-config configs/training/default.yaml \
        --output-dir finetuned_models/qwen2-audio-lora \
        --datasets crema_asis crema_d meld

    python scripts/finetune.py \
        --model audio-flamingo3 \
        --model-config configs/models/audio_flamingo3.yaml \
        --train-config configs/training/default.yaml \
        --output-dir finetuned_models/audio-flamingo3-lora \
        --datasets crema_asis crema_d meld

    # NOTE: kimi-audio requires deepspeed to run multi-gpu fine-tuning
    deepspeed --num_gpus=2 scripts/finetune.py \
        --model kimi-audio \
        --model-config configs/models/kimi_audio.yaml \
        --train-config configs/training/training_kimi_audio.yaml \
        --output-dir finetuned_models/kimi-audio-lora \
        --datasets crema_asis crema_d meld
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

from src.data.datasets import (
    CREMA_ASIS,
    CREMA_D,
    MELD_TRAIN,
    MELD_VAL,
    DatasetConfig,
    load_and_merge_datasets,
)
from src.training.finetune import run_finetuning
from src.utils.seed import set_seed


# Shorthand names → DatasetConfig lists
#   crema_asis : the synthesized, annotated dataset (this repo)
#   crema_d    : the original CREMA-D corpus
DATASET_PRESETS = {
    "crema_asis": [CREMA_ASIS],
    "crema_d": [CREMA_D],
    "meld": [MELD_TRAIN, MELD_VAL],
}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument(
        "--model", required=True,
        choices=["qwen2-audio", "kimi-audio", "audio-flamingo3"],
    )
    parser.add_argument("--model-config", required=True, help="Model YAML config")
    parser.add_argument("--train-config", required=True, help="Training YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--datasets", nargs="+", default=["crema_asis", "crema_d", "meld"],
        help="Dataset presets to combine",
    )
    parser.add_argument("--prompt", default="configs/prompts/emotion_sentiment.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args, _ = parser.parse_known_args()

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

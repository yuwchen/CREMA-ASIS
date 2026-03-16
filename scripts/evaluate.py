#!/usr/bin/env python3
"""Step 3 and 4: Evaluate base or fine-tuned models on a dataset.

Usage (base model):
    python scripts/evaluate.py \
        --model qwen2-audio \
        --model-config configs/models/qwen2_audio.yaml \
        --data data/test.csv \
        --output results/qwen2_base.csv

Usage (fine-tuned with LoRA):
    python scripts/evaluate.py \
        --model qwen2-audio \
        --model-config configs/models/qwen2_audio.yaml \
        --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
        --data data/test.csv \
        --output results/qwen2_lora.csv

Usage (parquet dataset like LISTEN):
    python scripts/evaluate.py \
        --model kimi-audio \
        --model-config configs/models/kimi_audio.yaml \
        --lora-path finetuned_models/kimi-audio-lora/checkpoint \
        --data data/LISTEN/test-00000-of-00001.parquet \
        --data-format parquet \
        --temp-audio-dir LISTEN_audios \
        --output results/kimi_lora_LISTEN.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.evaluation.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="Model inference / evaluation")
    parser.add_argument(
        "--model", required=True,
        choices=["qwen2-audio", "kimi-audio", "audio-flamingo3"],
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data", required=True, help="CSV or Parquet file")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--lora-path", default=None, help="LoRA checkpoint")
    parser.add_argument("--prompt", default="configs/prompts/emotion_sentiment.txt")
    parser.add_argument("--data-format", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--audio-column", default="filepath")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temp-audio-dir", default=None)
    args = parser.parse_args()

    run_inference(
        model_type=args.model,
        model_config_path=args.model_config,
        prompt_path=args.prompt,
        data_source=args.data,
        output_csv=args.output,
        lora_path=args.lora_path,
        data_format=args.data_format,
        audio_column=args.audio_column,
        id_column=args.id_column,
        sr=args.sr,
        device=args.device,
        temp_audio_dir=args.temp_audio_dir,
    )


if __name__ == "__main__":
    main()

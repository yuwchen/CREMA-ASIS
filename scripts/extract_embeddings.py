#!/usr/bin/env python3
"""Step 5: Extract and cache layer-wise embeddings.

Supports all three models and their sub-components (LLM, Whisper, etc.).

Usage (Qwen2-Audio LLM layers):
    python scripts/extract_embeddings.py \
        --model qwen2-audio --component llm \
        --model-config configs/models/qwen2_audio.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/

Usage (Qwen2-Audio Whisper encoder):
    python scripts/extract_embeddings.py \
        --model qwen2-audio --component whisper \
        --model-config configs/models/qwen2_audio.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/

Usage (Kimi-Audio with LoRA):
    python scripts/extract_embeddings.py \
        --model kimi-audio --component llm \
        --model-config configs/models/kimi_audio.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/ \
        --lora-path finetuned_models/kimi-lora/checkpoint-7500

Usage (Audio-Flamingo3):
    python scripts/extract_embeddings.py \
        --model audio-flamingo3 --component llm \
        --model-config configs/models/audio_flamingo3.yaml \
        --csv data/cremad_all_clean_w_sad_filtered.csv \
        --data-dir data/cremad-sync/cremad-sync-wsad/
"""

import argparse
import gc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import torch

from src.data.datasets import load_data_for_probing
from src.extraction.cache import extract_and_cache_all, get_cache_path
from src.utils.io import load_yaml, load_prompt


def _build_extract_fn(model_type, component, audio_model, selected_layers, prompt,
                      use_float16, device):
    """Return a callable ``extract_fn(audio_path) -> dict``."""

    if model_type == "qwen2-audio" and component == "llm":
        def fn(path):
            return audio_model.extract_llm_embeddings(
                path, selected_layers, prompt, use_float16=use_float16, device=device,
            )
        return fn

    elif model_type == "qwen2-audio" and component == "whisper":
        def fn(path):
            return audio_model.extract_whisper_embeddings(
                path, selected_layers, prompt, use_float16=use_float16, device=device,
            )
        return fn

    elif model_type == "kimi-audio":
        from src.models.kimi_audio import KimiAudioEmbeddingExtractor
        extractor = KimiAudioEmbeddingExtractor(
            model_path=audio_model.model_name,
            selected_layers=selected_layers,
        )
        # Optional LoRA
        return lambda path: extractor.extract_file(path, prompt, use_float16=use_float16)

    elif model_type == "audio-flamingo3":
        from src.models.audio_flamingo3 import AudioFlamingoEmbeddingExtractor
        extractor = AudioFlamingoEmbeddingExtractor(
            audio_model.model, audio_model.processor, selected_layers,
        )
        return lambda path: extractor.extract_file(path, prompt, use_float16=use_float16)

    else:
        raise ValueError(f"Unknown model_type/component: {model_type}/{component}")


def main():
    parser = argparse.ArgumentParser(description="Extract & cache embeddings")
    parser.add_argument(
        "--model", required=True,
        choices=["qwen2-audio", "kimi-audio", "audio-flamingo3"],
    )
    parser.add_argument("--component", default="llm", choices=["llm", "whisper"])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--csv", required=True, help="Data CSV path")
    parser.add_argument("--data-dir", required=True, help="Audio file directory")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--prompt", default="configs/prompts/emotion_sentiment.txt")
    parser.add_argument("--cache-dir", default="embedding_cache")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--memory-efficient", action="store_true", default=True)
    parser.add_argument("--sample-pct", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--not-float16", action="store_true")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    prompt = load_prompt(args.prompt)
    use_float16 = not args.not_float16
    use_lora = args.lora_path is not None

    # -- Determine selected layers from config --
    comp_cfg = model_cfg["extraction"].get(args.component, {})
    selected_layers = comp_cfg.get("selected_layers", [0, 1, 6, 13, 20, 27])

    # -- Load data splits --
    data = load_data_for_probing(args.csv, args.data_dir, args.sample_pct)

    # -- Load model (for qwen2/flamingo; kimi loads inside extractor) --
    audio_model = None
    if args.model in ("qwen2-audio", "audio-flamingo3"):
        from src.models import get_model
        audio_model = get_model(args.model, model_name=model_cfg["model_name"])
        audio_model.load(device=args.device, lora_path=args.lora_path)
    else:
        from src.models.kimi_audio import KimiAudioModel
        audio_model = KimiAudioModel(model_name=model_cfg["model_name"])
        # Kimi extractor loads its own model internally

    # -- Build extract function --
    extract_fn = _build_extract_fn(
        args.model, args.component, audio_model, selected_layers,
        prompt, use_float16, args.device,
    )

    # -- Extract for each split --
    data_id = os.path.basename(args.csv).replace(".csv", "")
    for split in ("train", "val", "test"):
        files = data[f"{split}_files"]
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
        print(f"\n{'=' * 60}")
        print(f"Split: {split} ({len(files)} files) → {cache_path}")
        print(f"{'=' * 60}")

        extract_and_cache_all(
            extract_fn=extract_fn,
            file_list=files,
            cache_path=cache_path,
            save_every_n=args.save_every,
            memory_efficient=args.memory_efficient,
        )

    # -- Cleanup --
    del audio_model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nExtraction complete for all splits.")


if __name__ == "__main__":
    main()

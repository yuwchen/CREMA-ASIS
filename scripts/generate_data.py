#!/usr/bin/env python3
"""Step 1: Generate synthetic audio samples using IndexTTS.

Usage:
    python scripts/generate_data.py \
        --cfg checkpoints/config.yaml \
        --model-dir checkpoints \
        --speaker ./CREMA-D/AudioWAV_en/1001_DFA_ANG_XX.wav \
        --text "I appreciate it, that's good to know." \
        --output ./samples/output.wav \
        --emotion 0 0 0 0 0 0 0 0
"""

import argparse

from src.data.generation import generate_sample


def main():
    parser = argparse.ArgumentParser(description="Generate TTS samples with IndexTTS")
    parser.add_argument("--cfg", required=True, help="Path to IndexTTS config YAML")
    parser.add_argument("--model-dir", required=True, help="IndexTTS checkpoint dir")
    parser.add_argument("--speaker", required=True, help="Reference speaker WAV")
    parser.add_argument("--text", required=True, help="Target sentence")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument(
        "--emotion", nargs=8, type=int, default=[0] * 8,
        help="8-dim emotion vector (default: all zeros)",
    )
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    generate_sample(
        cfg_path=args.cfg,
        model_dir=args.model_dir,
        speaker_audio=args.speaker,
        text=args.text,
        output_path=args.output,
        emotion_vector=args.emotion,
        use_fp16=args.fp16,
    )
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 2: Acoustic quality filtering on generated audio.

Usage:
    python scripts/filter_data.py \
        --wav-dir ./samples/ \
        --output-csv Results/acoustic_detection.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.data.filtering import filter_directory


def main():
    parser = argparse.ArgumentParser(description="Acoustic emotion filtering")
    parser.add_argument("--wav-dir", required=True, help="Directory of WAV files")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--model-id",
        default="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
        help="HuggingFace model for classification",
    )
    parser.add_argument(
        "--kept-csv", default=None,
        help="Optional path for a CSV holding only the retained samples",
    )
    parser.add_argument(
        "--no-policy", action="store_true",
        help="Only write AER predictions; skip the retention policy",
    )
    args = parser.parse_args()

    filter_directory(
        args.wav_dir,
        args.output_csv,
        model_id=args.model_id,
        apply_policy=not args.no_policy,
        kept_csv=args.kept_csv,
    )


if __name__ == "__main__":
    main()

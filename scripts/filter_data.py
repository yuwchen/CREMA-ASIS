#!/usr/bin/env python3
"""Step 1b: Acoustic quality filtering on generated audio.

Usage:
    python scripts/filter_data.py \
        --wav-dir ./samples/ \
        --output-csv Results/acoustic_detection.csv
"""

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
    args = parser.parse_args()

    filter_directory(args.wav_dir, args.output_csv, model_id=args.model_id)


if __name__ == "__main__":
    main()

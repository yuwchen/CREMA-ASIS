#!/usr/bin/env python3
"""Step 1 (batch): Generate synthetic audio samples from a CSV manifest.

Reads rows from a CSV file and generates one WAV per row, loading IndexTTS2
only once for efficiency.

Expected CSV columns:
  audio_path   - speaker reference filename (joined with --speaker-dir)
  output_text  - sentence to synthesise
  emo_vector   - 8-dim emotion vector as a Python list literal, e.g. "[0,0,0,0,0,0,0,0]"
  output_name  - output filename (written under --output-dir)

Usage:
    python scripts/generate_data_from_csv.py \
        --cfg checkpoints/config.yaml \
        --model-dir checkpoints \
        --speaker-dir ./CREMA-D/AudioWAV_en \
        --csv CREMA-ASIS-meta.csv \
        --output-dir ./samples \
        [--skip-existing] [--fp16]
"""

import argparse
import ast
import os
import pathlib
import sys

import pandas as pd

from src.data.generation import infer_sample


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate TTS samples from a CSV manifest"
    )
    parser.add_argument("--cfg", required=True, help="Path to IndexTTS config YAML")
    parser.add_argument("--model-dir", required=True, help="IndexTTS checkpoint dir")
    parser.add_argument(
        "--speaker-dir",
        required=True,
        help="Directory containing speaker reference WAVs (audio_path column)",
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV manifest")
    parser.add_argument("--output-dir", required=True, help="Directory for output WAVs")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose output file already exists",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="Device for inference (e.g. cuda:0, cuda:1, cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    _third_party = pathlib.Path(__file__).resolve().parents[1] / "third_party"
    if str(_third_party) not in sys.path:
        sys.path.insert(0, str(_third_party))
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.cfg,
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        use_cuda_kernel=False,
        use_deepspeed=False,
        device=args.device,
    )

    total = len(df)
    for i, row in df.iterrows():
        output_path = os.path.join(args.output_dir, row["output_name"])

        if args.skip_existing and os.path.exists(output_path):
            print(f"[{i+1}/{total}] Skipping (exists): {output_path}")
            continue

        speaker_audio = os.path.join(args.speaker_dir, row["audio_path"])
        text = row["output_text"]
        emotion_vector = ast.literal_eval(row["emo_vector"])

        print(f"[{i+1}/{total}] {row['output_name']}")
        infer_sample(tts, speaker_audio, text, output_path, emotion_vector)

    print(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

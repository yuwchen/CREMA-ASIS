#!/usr/bin/env python3
"""Step 2b: Transcribe generated speech and compute word error rate.

Two uses:

1. Verify the TTS renders the intended sentence.  Adds
   ``transcript`` and ``wer`` columns to the manifest; the test-set
   WER < 0.5 cut is then applied by scripts/evaluate.py.
2. Score a LALM's own transcripts to check SFT did not hurt ASR, 
   via ``--score-only``.

Usage (transcribe generated audio and add a wer column):
    python scripts/compute_wer.py \
        --csv data/CREMA-ASIS_meta.csv \
        --data-dir data/crema-asis/cremad-sync-wsad \
        --output data/CREMA-ASIS_meta_wer.csv

Usage (score transcripts that are already in a CSV):
    python scripts/compute_wer.py \
        --csv results/qwen2_lora_transcription.csv \
        --score-only \
        --text-column output_text \
        --transcript-column model_transcript \
        --output results/qwen2_lora_wer.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from src.data.transcription import (
    DEFAULT_ASR_MODEL,
    load_asr,
    transcribe_dataframe,
    word_error_rate,
)


def main():
    parser = argparse.ArgumentParser(description="Transcription + WER")
    parser.add_argument("--csv", required=True, help="Input manifest CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--data-dir", default=None, help="Directory with the WAV files")
    parser.add_argument("--audio-column", default="output_name")
    parser.add_argument("--text-column", default="output_text")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--device", default=None, help="0 for cuda:0, -1 for cpu")
    parser.add_argument(
        "--score-only", action="store_true",
        help="Skip ASR; score an existing transcript column",
    )
    parser.add_argument("--transcript-column", default="transcript")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    if args.score_only:
        df["wer"] = [
            word_error_rate(r[args.text_column], r[args.transcript_column])
            for _, r in df.iterrows()
        ]
    else:
        if not args.data_dir:
            parser.error("--data-dir is required unless --score-only is given")
        asr = load_asr(
            args.asr_model,
            device=int(args.device) if args.device is not None else None,
        )
        df = transcribe_dataframe(
            df, args.data_dir, asr,
            audio_column=args.audio_column, text_column=args.text_column,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nMean WER : {df['wer'].mean():.4f}")
    print(f"WER < 0.5: {(df['wer'] < 0.5).sum()}/{len(df)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 4c: Build the out-of-domain LISTEN evaluation subsets.

Drops CREMA-D / MELD samples, splits the remaining rows into the acoustic and
semantic tasks by question text, and normalises each task's labels. Writes one
parquet per task, ready for scripts/evaluate.py --data-format parquet.

Download the source split first:
    https://huggingface.co/datasets/VibeCheck1/LISTEN_full
    data/test-00000-of-00001.parquet

Usage:
    python scripts/prepare_listen.py \
        --parquet data/LISTEN/test-00000-of-00001.parquet \
        --output-dir data/LISTEN

Inspect the schema and label distribution without writing anything:
    python scripts/prepare_listen.py \
        --parquet data/LISTEN/test-00000-of-00001.parquet --inspect

Then evaluate and score:
    python scripts/evaluate.py \
        --model qwen2-audio --model-config configs/models/qwen2_audio.yaml \
        --data data/LISTEN/listen_acoustic.parquet --data-format parquet \
        --temp-audio-dir LISTEN_audios \
        --output results/qwen2_base_listen_acoustic.csv

    python scripts/compute_metrics.py \
        --predictions results/qwen2_base_listen_acoustic.csv \
        --single-task --true-column acoustic --pred-column acoustic_emotion \
        --label-map configs/data/listen_label_map.yaml \
        --label-map-key acoustic_synonyms
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from src.data.listen import prepare_listen
from src.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Prepare the LISTEN test split")
    parser.add_argument("--parquet", required=True, help="LISTEN test parquet")
    parser.add_argument("--output-dir", default="data/LISTEN")
    parser.add_argument("--config", default="configs/data/listen_label_map.yaml")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print the schema and label counts, then exit without writing",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    if args.inspect:
        print(f"{len(df)} rows, {len(df.columns)} columns\n")
        for col in df.columns:
            if col.lower() in ("audio", "bytes"):
                print(f"  {col}: <audio>")
                continue
            uniq = df[col].nunique(dropna=True)
            preview = list(df[col].dropna().unique()[:12]) if uniq <= 40 else "..."
            print(f"  {col}: {uniq} unique  {preview}")
        return

    label_map = load_yaml(args.config)
    acoustic_df, semantic_df, report = prepare_listen(
        df, label_map,
        id_column=args.id_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
    )

    print(f"Input rows              : {report['n_input']}")
    print(f"Removed (CREMA-D/MELD)  : {report['n_source_excluded']}")
    print(f"Acoustic task rows      : {report['n_acoustic_rows']}")
    print(f"  kept                  : {report['n_acoustic']} "
          f"(dropped {report['n_acoustic_dropped']} out-of-vocabulary)")
    print(f"Semantic task rows      : {report['n_semantic_rows']}")
    print(f"  kept                  : {report['n_semantic']} "
          f"(dropped {report['n_semantic_dropped']} out-of-vocabulary)")

    for task in ("acoustic", "semantic"):
        dropped = report.get(f"{task}_dropped_labels")
        if dropped:
            print(f"\nDropped {task} labels (no entry in the config):")
            print(f"  {dropped}")
            print(f"  Add them to configs/data/listen_label_map.yaml to keep them.")

    if report.get("unmatched_questions"):
        print("\nWARNING: questions matching neither task prompt list:")
        for q in report["unmatched_questions"][:10]:
            print(f"  {q!r}")
        print("  Rows with these questions are in neither subset.")

    os.makedirs(args.output_dir, exist_ok=True)
    acoustic_path = os.path.join(args.output_dir, "listen_acoustic.parquet")
    semantic_path = os.path.join(args.output_dir, "listen_semantic.parquet")
    acoustic_df.to_parquet(acoustic_path, index=False)
    semantic_df.to_parquet(semantic_path, index=False)
    print(f"\nSaved {acoustic_path}\n      {semantic_path}")


if __name__ == "__main__":
    main()

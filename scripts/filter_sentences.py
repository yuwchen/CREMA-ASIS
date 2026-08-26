#!/usr/bin/env python3
"""Step 0: GPT-4o filtering of candidate GoEmotions sentences.

Runs before audio generation.  Drops sentences that do not carry the target
sentiment, are offensive, or would never be said aloud.

Requires an OpenAI API key:
    export OPENAI_API_KEY=...

Usage:
    python scripts/filter_sentences.py \
        --csv data/goemotions_candidates.csv \
        --sentence-column sentence \
        --sentiment-column sentiment \
        --output data/goemotions_filtered.csv

With one fixed target sentiment for the whole file:
    python scripts/filter_sentences.py \
        --csv data/goemotions_positive.csv \
        --sentiment positive \
        --output data/goemotions_positive_filtered.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from src.data.sentence_filtering import DEFAULT_MODEL, filter_sentences
from src.utils.io import load_prompt


def main():
    parser = argparse.ArgumentParser(description="GPT-4o sentence selection")
    parser.add_argument("--csv", required=True, help="Candidate sentences CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--sentence-column", default="sentence")
    parser.add_argument(
        "--sentiment-column", default="sentiment",
        help="Per-row target sentiment column (ignored if --sentiment is given)",
    )
    parser.add_argument(
        "--sentiment", default=None,
        help="Fixed target sentiment for every row: positive | negative | neutral",
    )
    parser.add_argument("--prompt", default="configs/prompts/sentence_selection.txt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--kept-csv", default=None,
        help="Optional path for a CSV holding only the retained sentences",
    )
    args = parser.parse_args()

    template = load_prompt(args.prompt)
    df = pd.read_csv(args.csv)

    result = filter_sentences(
        df,
        template=template,
        sentence_column=args.sentence_column,
        sentiment_column=None if args.sentiment else args.sentiment_column,
        sentiment=args.sentiment,
        model=args.model,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)

    n_keep = int(result["keep"].sum())
    print(f"\nKept {n_keep}/{len(result)} sentences ({n_keep / max(len(result), 1):.1%})")
    if (result["gpt_rejected"] == True).any():  # noqa: E712
        counts = (
            result.loc[result["gpt_rejected"] == True, "gpt_reasons"]  # noqa: E712
            .str.split(",").explode().value_counts()
        )
        print("Rejection reasons:")
        for reason, n in counts.items():
            if reason:
                print(f"  criterion {reason}: {n}")
    print(f"Saved to {args.output}")

    if args.kept_csv:
        os.makedirs(os.path.dirname(args.kept_csv) or ".", exist_ok=True)
        result[result["keep"]].to_csv(args.kept_csv, index=False)
        print(f"Retained subset saved to {args.kept_csv}")


if __name__ == "__main__":
    main()

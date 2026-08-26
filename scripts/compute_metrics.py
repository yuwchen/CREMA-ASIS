#!/usr/bin/env python3
"""Step 4b: Score the prediction CSVs written by scripts/evaluate.py.

Reports Acc_acou / Acc_sem / Acc_dual and, with
``--per-condition``, precision / recall / F1 for every acoustic-semantic
pair.

Usage:
    python scripts/compute_metrics.py \
        --predictions results/qwen2_base.csv

    python scripts/compute_metrics.py \
        --predictions results/qwen2_base.csv results/qwen2_lora.csv \
        --per-condition \
        --output results/metrics_summary.csv

Single-task runs (the joint MELD / out-of-domain LISTEN settings):
    python scripts/compute_metrics.py \
        --predictions results/qwen2_lora_meld.csv \
        --single-task --true-column Emotion --pred-column acoustic_emotion

    python scripts/compute_metrics.py \
        --predictions results/qwen2_lora_listen_acoustic.csv \
        --single-task --true-column acoustic --pred-column acoustic_emotion \
        --label-map configs/data/listen_label_map.yaml \
        --label-map-key acoustic_synonyms

"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from src.utils.io import load_yaml
from src.evaluation.metrics import (
    load_predictions,
    overall_accuracy,
    per_condition_prf,
    score_listen,
    single_task_accuracy,
)


def main():
    parser = argparse.ArgumentParser(description="Score CREMA-ASIS predictions")
    parser.add_argument(
        "--predictions", required=True, nargs="+",
        help="One or more prediction CSVs from scripts/evaluate.py",
    )
    parser.add_argument(
        "--per-condition", action="store_true",
        help="Also report precision/recall/F1 per acoustic-semantic pair",
    )
    parser.add_argument("--output", default=None, help="Write the summary to this CSV")
    parser.add_argument("--acoustic-true", default="acoustic")
    parser.add_argument("--semantic-true", default="semantic")
    parser.add_argument("--acoustic-pred", default="acoustic_emotion")
    parser.add_argument("--semantic-pred", default="semantic_sentiment")
    parser.add_argument(
        "--single-task", action="store_true",
        help="Score one label column instead of the dual acoustic/semantic task",
    )
    parser.add_argument(
        "--listen", action="store_true",
        help="Score a prediction CSV covering the whole LISTEN split: split rows "
             "into the acoustic and semantic tasks by question text, then score "
             "both. Uses --label-map (defaults to the LISTEN config).",
    )
    parser.add_argument("--true-column", default=None, help="--single-task gold column")
    parser.add_argument("--pred-column", default=None, help="--single-task prediction column")
    parser.add_argument(
        "--label-map", default=None,
        help="YAML holding a synonym map, applied to both gold and predicted "
             "labels so the two sides share one vocabulary (--single-task only)",
    )
    parser.add_argument(
        "--label-map-key", default=None,
        help="Key to read from --label-map, e.g. acoustic_synonyms",
    )
    args = parser.parse_args()

    label_map = None
    if args.label_map:
        loaded = load_yaml(args.label_map)
        label_map = loaded.get(args.label_map_key) if args.label_map_key else loaded
        if not isinstance(label_map, dict):
            parser.error(
                f"--label-map-key '{args.label_map_key}' did not resolve to a "
                f"mapping in {args.label_map}"
            )

    summary = []

    for path in args.predictions:
        name = os.path.basename(path).replace(".csv", "")

        if args.listen:
            cfg = load_yaml(args.label_map or "configs/data/listen_label_map.yaml")
            res = score_listen(path, cfg)
            rep = res["report"]
            print(f"\n{name}")
            print(f"  rows                  : {rep['n_input']}")
            print(f"  removed (CREMA-D/MELD): {rep['n_source_excluded']}")
            for task in ("acoustic", "semantic"):
                t = res[task]
                print(f"  {task.capitalize():9s} n={t['n']:5d} "
                      f"unparsed={t['n_unparsed']:4d}  Acc={t['accuracy']:.3f}"
                      f"   (dropped {rep[f'n_{task}_dropped']} out-of-vocabulary)")
            summary.append({
                "run": name,
                "n_acoustic": res["acoustic"]["n"],
                "acc_acoustic": res["acoustic"]["accuracy"],
                "n_semantic": res["semantic"]["n"],
                "acc_semantic": res["semantic"]["accuracy"],
            })
            continue

        if args.single_task:
            if not (args.true_column and args.pred_column):
                parser.error("--single-task requires --true-column and --pred-column")
            res = single_task_accuracy(
                path, args.true_column, args.pred_column, label_map=label_map,
            )
            print(f"\n{name}  (n={res['n']}, unparsed={res['n_unparsed']})")
            print(f"  Accuracy : {res['accuracy']:.3f}")
            summary.append({"run": name, **res})
            continue

        df = load_predictions(
            path,
            acoustic_true=args.acoustic_true, semantic_true=args.semantic_true,
            acoustic_pred=args.acoustic_pred, semantic_pred=args.semantic_pred,
        )
        res = overall_accuracy(df)
        print(f"\n{name}  (n={res['n']}, unparsed={res['n_unparsed']})")
        print(f"  Acc_acou : {res['acc_acoustic']:.3f}")
        print(f"  Acc_sem  : {res['acc_semantic']:.3f}")
        print(f"  Acc_dual : {res['acc_dual']:.3f}")
        summary.append({"run": name, **res})

        if args.per_condition:
            prf = per_condition_prf(df)
            print()
            print(prf.to_string(
                index=False,
                formatters={
                    "precision": "{:.3f}".format,
                    "recall": "{:.3f}".format,
                    "f1": "{:.3f}".format,
                },
            ))
            if args.output:
                prf_path = args.output.replace(".csv", f"_{name}_per_condition.csv")
                prf.to_csv(prf_path, index=False)
                print(f"\n  per-condition table -> {prf_path}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        pd.DataFrame(summary).to_csv(args.output, index=False)
        print(f"\nSummary -> {args.output}")


if __name__ == "__main__":
    main()

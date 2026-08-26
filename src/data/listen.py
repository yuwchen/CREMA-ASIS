"""Preprocessing for the out-of-domain LISTEN evaluation.

LISTEN_full is long-format: one row per (sample, question), with columns
``id``, ``question``, and ``answer``. Which modality a row belongs to is
encoded in the question text, not in a separate column, so the split is made
by matching ``question`` against the fixed prompt lists in
``configs/data/listen_label_map.yaml``.

Three steps before the data is usable:

1. drop samples whose ``id`` names CREMA-D or MELD, which are in our SFT data,
2. split the rest into the acoustic and semantic tasks by question text,
3. normalise each task's answers and drop labels with no clean counterpart.

The acoustic task keeps the eight categories the LALM prompt offers
(including fear, surprise, and calm), not the five in CREMA-ASIS -- the model
is scored on the label space it was asked to choose from. The semantic task
collapses LISTEN's emotion words onto the three sentiment polarities.

Label matching is exact and case-sensitive, mirroring the dictionary lookups
in the original evaluation script.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def _apply_synonyms(value, synonyms: Dict[str, str]):
    """Normalise one label through the synonym map, leaving it as-is if absent."""
    return synonyms.get(value, value)


def prepare_listen(
    df: pd.DataFrame,
    label_map: Dict,
    id_column: str = "id",
    question_column: str = "question",
    answer_column: str = "answer",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Split LISTEN into the acoustic and semantic evaluation subsets.

    Args:
        df: Raw LISTEN dataframe.
        label_map: Parsed ``listen_label_map.yaml``.
        id_column: Column holding the sample id (carries the corpus name).
        question_column: Column holding the prompt text.
        answer_column: Column holding the ground-truth label.

    Returns:
        ``(acoustic_df, semantic_df, report)``. The acoustic subset gains an
        ``acoustic`` column and the semantic subset a ``semantic`` column,
        each holding the normalised ground-truth label.
    """
    report: Dict = {"n_input": len(df)}

    for col in (id_column, question_column, answer_column):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. LISTEN_full is expected to have "
                f"id/question/answer. Available: {list(df.columns)}"
            )

    # -- 1. Drop corpora that overlap with our fine-tuning data --
    substrings = list(label_map.get("exclude_id_substrings", []))
    excluded = df[id_column].map(
        lambda v: any(s in str(v) for s in substrings)
    )
    report["n_source_excluded"] = int(excluded.sum())
    df = df[~excluded]

    acoustic_qs = set(label_map.get("acoustic_questions", []))
    semantic_qs = set(label_map.get("semantic_questions", []))

    unmatched = sorted(
        set(df[question_column]) - acoustic_qs - semantic_qs
    )
    report["unmatched_questions"] = unmatched

    # -- 2 & 3. Split by question, then normalise and filter each task --
    def _subset(questions, out_name, synonyms, mapping, allowed) -> pd.DataFrame:
        rows = df[df[question_column].isin(questions)].copy()
        report[f"n_{out_name}_rows"] = int(len(rows))

        labels = rows[answer_column]
        if synonyms:
            labels = labels.map(lambda v: _apply_synonyms(v, synonyms))
        if mapping:
            labels = labels.map(lambda v: mapping.get(v, v))

        keep = labels.isin(allowed)
        report[f"{out_name}_dropped_labels"] = sorted(
            set(labels[~keep].astype(str))
        )
        report[f"n_{out_name}_dropped"] = int((~keep).sum())

        out = rows[keep].copy()
        out[out_name] = labels[keep].values
        return out

    acoustic_df = _subset(
        acoustic_qs, "acoustic",
        label_map.get("acoustic_synonyms") or {},
        None,
        set(label_map.get("acoustic_labels", [])),
    )
    semantic_df = _subset(
        semantic_qs, "semantic",
        None,
        label_map.get("semantic_map") or {},
        set(label_map.get("semantic_labels", [])),
    )

    report["n_acoustic"] = len(acoustic_df)
    report["n_semantic"] = len(semantic_df)
    return acoustic_df, semantic_df, report

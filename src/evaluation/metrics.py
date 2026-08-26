"""Scoring for CREMA-ASIS evaluation outputs.

Consumes the prediction CSVs written by ``scripts/evaluate.py`` and produces
the numbers reported in the paper:

- ``Acc_acou``, ``Acc_sem``, ``Acc_dual``
- per-condition precision / recall / F1 over acoustic-semantic pairs

Samples the model failed to process are left with empty predictions by
``run_inference`` and are counted as incorrect, matching the paper's protocol.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# Grouping of acoustic-semantic pairs into the reported conditions.
INCONGRUOUS = {
    ("disgust", "positive"), ("angry", "positive"),
    ("sad", "positive"), ("happy", "negative"),
}
CONGRUOUS = {
    ("disgust", "negative"), ("angry", "negative"),
    ("sad", "negative"), ("happy", "positive"),
}


def _norm(series: pd.Series) -> pd.Series:
    """Lower-case, strip, and turn missing values into the empty string."""
    return series.fillna("").astype(str).str.strip().str.lower()


def condition_group(acoustic: str, semantic: str) -> str:
    """Return ``incongruous`` / ``congruous`` / ``neutral-associated``."""
    if (acoustic, semantic) in INCONGRUOUS:
        return "incongruous"
    if (acoustic, semantic) in CONGRUOUS:
        return "congruous"
    return "neutral-associated"


def load_predictions(
    csv_path: str,
    acoustic_true: str = "acoustic",
    semantic_true: str = "semantic",
    acoustic_pred: str = "acoustic_emotion",
    semantic_pred: str = "semantic_sentiment",
) -> pd.DataFrame:
    """Load a prediction CSV and normalise the four label columns."""
    df = pd.read_csv(csv_path, low_memory=False)
    wanted = {
        "--acoustic-true": acoustic_true,
        "--semantic-true": semantic_true,
        "--acoustic-pred": acoustic_pred,
        "--semantic-pred": semantic_pred,
    }
    missing = {flag: col for flag, col in wanted.items() if col not in df.columns}
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing.values())}\n"
            f"  Columns in the file: {list(df.columns)}\n"
            f"  Point at the right ones with: "
            + " ".join(f"{flag} <col>" for flag in missing)
        )

    out = pd.DataFrame({
        "acoustic_true": _norm(df[acoustic_true]),
        "semantic_true": _norm(df[semantic_true]),
        "acoustic_pred": _norm(df[acoustic_pred]),
        "semantic_pred": _norm(df[semantic_pred]),
    })
    out["acoustic_correct"] = out["acoustic_true"] == out["acoustic_pred"]
    out["semantic_correct"] = out["semantic_true"] == out["semantic_pred"]
    out["dual_correct"] = out["acoustic_correct"] & out["semantic_correct"]
    return out


def overall_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    """Return ``Acc_acou`` / ``Acc_sem`` / ``Acc_dual`` plus bookkeeping counts."""
    n_unparsed = int(((df["acoustic_pred"] == "") | (df["semantic_pred"] == "")).sum())
    return {
        "n": int(len(df)),
        "n_unparsed": n_unparsed,
        "acc_acoustic": float(df["acoustic_correct"].mean()),
        "acc_semantic": float(df["semantic_correct"].mean()),
        "acc_dual": float(df["dual_correct"].mean()),
    }


def per_condition_prf(df: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 for every acoustic-semantic pair.

    A sample counts as a positive prediction for a pair when the model
    predicts *both* that acoustic emotion and that semantic sentiment.
    """
    true_pair = list(zip(df["acoustic_true"], df["semantic_true"]))
    pred_pair = list(zip(df["acoustic_pred"], df["semantic_pred"]))

    rows: List[dict] = []
    for pair in sorted(set(true_pair)):
        tp = sum(t == pair and p == pair for t, p in zip(true_pair, pred_pair))
        fp = sum(t != pair and p == pair for t, p in zip(true_pair, pred_pair))
        fn = sum(t == pair and p != pair for t, p in zip(true_pair, pred_pair))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append({
            "acoustic": pair[0],
            "semantic": pair[1],
            "group": condition_group(*pair),
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    order = {"incongruous": 0, "congruous": 1, "neutral-associated": 2}
    return (
        pd.DataFrame(rows)
        .sort_values(["group", "acoustic", "semantic"], key=lambda s: s.map(order).fillna(s))
        .reset_index(drop=True)
    )


def score_listen(
    csv_path: str,
    label_map: Dict,
    id_column: str = "id",
    question_column: str = "question",
    answer_column: str = "answer",
    acoustic_pred: str = "acoustic_emotion",
    semantic_pred: str = "semantic_sentiment",
) -> Dict[str, Dict]:
    """Score a prediction CSV that covers the whole LISTEN split.

    Use this when a model was run over the entire LISTEN parquet in one pass,
    so the file still carries ``id`` / ``question`` / ``answer`` alongside the
    predictions.  The rows are split into the acoustic and semantic tasks by
    question text, filtered and normalised exactly as
    :func:`src.data.listen.prepare_listen` does, then scored.

    (When the subsets were built ahead of time with ``prepare_listen.py``,
    score them with :func:`single_task_accuracy` instead.)

    Args:
        csv_path: Prediction CSV.
        label_map: Parsed ``listen_label_map.yaml``.
        id_column / question_column / answer_column: LISTEN's own columns.
        acoustic_pred / semantic_pred: The model's prediction columns.

    Returns:
        ``{"acoustic": {...}, "semantic": {...}, "report": {...}}`` where each
        task dict holds ``n``, ``n_unparsed``, and ``accuracy``.
    """
    from src.data.listen import prepare_listen

    df = pd.read_csv(csv_path, low_memory=False)
    acoustic_df, semantic_df, report = prepare_listen(
        df, label_map,
        id_column=id_column,
        question_column=question_column,
        answer_column=answer_column,
    )

    # The synonym map is applied to predictions too, so both sides of the
    # comparison share one vocabulary.
    synonyms = {
        str(k).lower(): str(v).lower()
        for k, v in (label_map.get("acoustic_synonyms") or {}).items()
    }

    def _score(sub, true_col, pred_col, mapping):
        if pred_col not in sub.columns:
            raise ValueError(
                f"{csv_path} has no '{pred_col}' column. "
                f"Columns in the file: {list(sub.columns)}"
            )
        if len(sub) == 0:
            return {"n": 0, "n_unparsed": 0, "accuracy": float("nan")}
        true = _norm(sub[true_col])
        pred = _norm(sub[pred_col])
        if mapping:
            pred = pred.map(lambda v: mapping.get(v, v))
        return {
            "n": int(len(sub)),
            "n_unparsed": int((pred == "").sum()),
            "accuracy": float((true == pred).mean()),
        }

    return {
        "acoustic": _score(acoustic_df, "acoustic", acoustic_pred, synonyms),
        "semantic": _score(semantic_df, "semantic", semantic_pred, None),
        "report": report,
    }


def single_task_accuracy(
    csv_path: str,
    true_column: str,
    pred_column: str,
    label_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Accuracy for a single-label run (the joint MELD / LISTEN settings).

    Args:
        csv_path: Prediction CSV.
        true_column: Gold label column.
        pred_column: Predicted label column.
        label_map: Optional normalisation applied to both sides after
            lower-casing (e.g. ``{"joy": "happy"}``).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    true = _norm(df[true_column])
    pred = _norm(df[pred_column])
    if label_map:
        mapping = {k.lower(): v.lower() for k, v in label_map.items()}
        true = true.map(lambda v: mapping.get(v, v))
        pred = pred.map(lambda v: mapping.get(v, v))
    keep = true != ""
    n_unparsed = int((pred[keep] == "").sum())
    return {
        "n": int(keep.sum()),
        "n_unparsed": n_unparsed,
        "accuracy": float((true[keep] == pred[keep]).mean()),
    }

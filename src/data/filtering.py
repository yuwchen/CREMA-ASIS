"""Acoustic emotion filtering using a pre-trained Whisper classifier.

Uses ``firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3``
to predict the acoustic emotion of each generated file, then applies the
retention policy that decides which samples enter CREMA-ASIS.

The AER model is imperfect, so a sample is kept when its prediction is either
the target emotion or a perceptually adjacent one; only clear mismatches are
dropped.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Set

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.utils.io import get_all_files
from src.utils.parsing import parse_filename


DEFAULT_MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

# Target acoustic emotion -> AER predictions that are accepted as a match.
# These are the pairs retained when CREMA-ASIS was built.
RETENTION_POLICY: Dict[str, Set[str]] = {
    "happy": {"happy", "surprised", "neutral"},
    "neutral": {"neutral", "angry"},
    "sad": {"neutral", "sad"},
    "disgust": {"surprised", "neutral"},
    "angry": {"angry", "surprised", "neutral"},
}


def is_retained(target_emotion: str, aer_prediction: str) -> bool:
    """Whether a sample survives acoustic filtering.

    Args:
        target_emotion: Intended acoustic emotion (from the reference clip).
        aer_prediction: Label predicted by the AER model.

    Returns:
        ``True`` when the pair is in :data:`RETENTION_POLICY`.  Targets absent
        from the policy are kept only on an exact match.
    """
    target = str(target_emotion).strip().lower()
    predicted = str(aer_prediction).strip().lower()
    return predicted in RETENTION_POLICY.get(target, {target})


def load_emotion_classifier(model_id: str = DEFAULT_MODEL_ID):
    """Load the emotion classification model and feature extractor.

    Returns:
        Tuple of ``(model, feature_extractor, id2label)``.
    """
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    return model, feature_extractor, id2label


def predict_emotion(
    audio_path: str,
    model,
    feature_extractor,
    id2label: dict,
    max_duration: float = 30.0,
) -> str:
    """Predict the acoustic emotion of a single audio file.

    Args:
        audio_path: Path to a WAV file.
        model: Pre-trained classification model.
        feature_extractor: Matching feature extractor.
        id2label: Mapping from class id to label string.
        max_duration: Maximum audio duration in seconds.

    Returns:
        Predicted emotion label string.
    """
    audio_array, _ = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[predicted_id]


def filter_directory(
    wav_dir: str,
    output_csv: str,
    model_id: str = DEFAULT_MODEL_ID,
    apply_policy: bool = True,
    kept_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run emotion prediction on all WAV files in a directory and filter them.

    The target emotion is read from the generated filename, which encodes it as
    ``<reference>-<acoustic>-<semantic>-<idx>.wav``.

    Args:
        wav_dir: Directory to scan for ``.wav`` files.
        output_csv: Path to write the full results CSV.
        model_id: HuggingFace model identifier.
        apply_policy: Whether to add ``target``/``keep`` columns by applying
            :func:`is_retained`.
        kept_csv: Optional path for a second CSV holding only retained rows.

    Returns:
        DataFrame with ``filepath`` and ``aer_prediction`` columns, plus
        ``target`` and ``keep`` when *apply_policy* is set.
    """
    model, feature_extractor, id2label = load_emotion_classifier(model_id)
    filelist = get_all_files(wav_dir, ".wav")

    results = []
    for filepath in tqdm(filelist, desc="Acoustic filtering"):
        try:
            prediction = predict_emotion(filepath, model, feature_extractor, id2label)
            row = {"filepath": filepath, "aer_prediction": prediction}

            if apply_policy:
                try:
                    target, _semantic = parse_filename(filepath)
                except ValueError:
                    target = None
                row["target"] = target
                row["keep"] = is_retained(target, prediction) if target else True

            results.append(row)
        except Exception as e:
            print(f"Error on {filepath}: {e}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} results to {output_csv}")

    if apply_policy and "keep" in df.columns:
        n_keep = int(df["keep"].sum())
        print(f"Retention policy: kept {n_keep}/{len(df)} "
              f"({n_keep / max(len(df), 1):.1%})")
        if kept_csv:
            os.makedirs(os.path.dirname(kept_csv) or ".", exist_ok=True)
            df[df["keep"]].to_csv(kept_csv, index=False)
            print(f"Retained subset saved to {kept_csv}")

    return df

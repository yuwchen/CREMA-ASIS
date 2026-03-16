"""Acoustic emotion filtering using a pre-trained Whisper classifier.

Uses ``firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3``
to predict the acoustic emotion of each audio file and produce a CSV of
results.  This can be used to quality-filter synthesised data.
"""

from __future__ import annotations

import os
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.utils.io import get_all_files


DEFAULT_MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"


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
) -> pd.DataFrame:
    """Run emotion prediction on all WAV files in a directory.

    Args:
        wav_dir: Directory to scan for ``.wav`` files.
        output_csv: Path to write results CSV.
        model_id: HuggingFace model identifier.

    Returns:
        DataFrame with ``filepath`` and ``emotion`` columns.
    """
    model, feature_extractor, id2label = load_emotion_classifier(model_id)
    filelist = get_all_files(wav_dir, ".wav")

    results = []
    for filepath in tqdm(filelist, desc="Acoustic filtering"):
        try:
            emotion = predict_emotion(filepath, model, feature_extractor, id2label)
            results.append({"filepath": filepath, "emotion": emotion})
        except Exception as e:
            print(f"Error on {filepath}: {e}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} results to {output_csv}")
    return df

"""Whisper transcription and word error rate for generated speech.

Used twice in the paper:

- Transcribe the TTS output with ``whisper-base.en`` and compare
  it against the sentence that was requested, to verify IndexTTS2 renders the
  intended text.  Test samples with WER > 0.5 are then excluded.
- Check that SFT does not degrade a LALM's transcription
  ability, by scoring its transcripts the same way.
"""

from __future__ import annotations

import os
import re
import string
from typing import List, Optional

import librosa
import pandas as pd
from tqdm import tqdm


DEFAULT_ASR_MODEL = "openai/whisper-base.en"

# Apostrophes are kept, so "thats" and "that's" count as different words.
_PUNCT = str.maketrans("", "", string.punctuation.replace("'", ""))


def normalise_text(text: str) -> List[str]:
    """Lower-case, strip punctuation, and split into words.

    Applied to both sides before scoring so that casing and punctuation
    differences between the target sentence and the ASR output are ignored.
    Apostrophes are preserved to match how the released ``wer`` column was
    computed.
    """
    text = str(text).lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate between *reference* and *hypothesis*.

    Standard Levenshtein distance over words divided by the reference length.
    Returns 0.0 when both sides are empty and 1.0 when only the reference is.
    """
    ref = normalise_text(reference)
    hyp = normalise_text(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    # Two-row dynamic programme over the edit-distance matrix.
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,                      # deletion
                current[j - 1] + 1,                   # insertion
                previous[j - 1] + (r != h),           # substitution
            )
        previous = current

    return previous[len(hyp)] / len(ref)


def load_asr(model_id: str = DEFAULT_ASR_MODEL, device: Optional[str] = None):
    """Load a Whisper ASR pipeline."""
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_id, device=device)


def transcribe_dataframe(
    df: pd.DataFrame,
    data_dir: str,
    asr,
    audio_column: str = "output_name",
    text_column: str = "output_text",
    sr: int = 16000,
) -> pd.DataFrame:
    """Transcribe every row and add ``transcript`` and ``wer`` columns.

    Args:
        df: Manifest with an audio filename column and a target text column.
        data_dir: Directory holding the audio files.
        asr: Pipeline returned by :func:`load_asr`.
        audio_column: Column with the audio filename.
        text_column: Column with the sentence that was requested.
        sr: Sampling rate for audio loading.

    Returns:
        A copy of *df* with ``transcript`` and ``wer`` columns.
    """
    transcripts, wers = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        path = os.path.join(data_dir, os.path.basename(str(row[audio_column])))
        try:
            audio, _ = librosa.load(path, sr=sr)
            hypothesis = asr(audio)["text"]
        except Exception as e:
            print(f"Error on {path}: {e}")
            hypothesis = ""
        transcripts.append(hypothesis)
        wers.append(word_error_rate(row[text_column], hypothesis))

    out = df.copy()
    out["transcript"] = transcripts
    out["wer"] = wers
    return out

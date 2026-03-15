"""Unified inference for base and fine-tuned models.

Consolidates ``qwen_inference.py``, ``flamingo_inference.py``, and
``kimi_re_run_listen_lora.py`` into a single evaluation loop.

Supports:
- Inference from WAV file paths
- Inference from parquet datasets (e.g. LISTEN) with in-memory audio bytes
- Automatic JSON response parsing with fallbacks
"""

from __future__ import annotations

import io
import json
import os
from typing import Optional

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.models import get_model
from src.utils.io import load_prompt, load_yaml
from src.utils.parsing import parse_emotion_response


def run_inference(
    model_type: str,
    model_config_path: str,
    prompt_path: str,
    data_source: str,
    output_csv: str,
    lora_path: Optional[str] = None,
    data_format: str = "csv",
    audio_column: str = "filepath",
    id_column: str = "id",
    sr: int = 16000,
    device: str = "auto",
    temp_audio_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run model inference on a dataset and save results.

    Args:
        model_type: ``"qwen2-audio"`` | ``"kimi-audio"`` | ``"audio-flamingo3"``
        model_config_path: Path to model YAML config.
        prompt_path: Path to prompt text file.
        data_source: Path to CSV or Parquet file.
        output_csv: Where to write the results CSV.
        lora_path: Optional LoRA checkpoint path.
        data_format: ``"csv"`` or ``"parquet"``.
        audio_column: Column with file paths (csv) or audio bytes (parquet).
        id_column: Column with sample identifiers.
        sr: Sampling rate for audio loading.
        device: Device for model loading.
        temp_audio_dir: Directory for temporary WAV files (parquet mode).

    Returns:
        DataFrame with inference results.
    """
    model_cfg = load_yaml(model_config_path)
    prompt = load_prompt(prompt_path)

    # -- Load model --
    audio_model = get_model(model_type, model_name=model_cfg["model_name"])
    audio_model.load(device=device, lora_path=lora_path)

    # -- Load data --
    if data_format == "parquet":
        df = pd.read_parquet(data_source)
    else:
        df = pd.read_csv(data_source)

    print(f"Running inference on {len(df)} samples with {model_type}")

    # -- Inference loop --
    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        new_row = row.to_dict()
        try:
            # Get audio path
            if data_format == "parquet" and isinstance(row.get("audio"), dict):
                # Parquet with embedded audio bytes
                audio_bytes = row["audio"]["bytes"]
                audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
                row_id = row.get(id_column, idx)
                if temp_audio_dir:
                    os.makedirs(temp_audio_dir, exist_ok=True)
                    temp_path = os.path.join(temp_audio_dir, f"{row_id}.wav")
                    sf.write(temp_path, audio, sr, subtype="PCM_16")
                    audio_path = temp_path
                else:
                    temp_path = f"/tmp/{row_id}.wav"
                    sf.write(temp_path, audio, sr, subtype="PCM_16")
                    audio_path = temp_path
            else:
                audio_path = row[audio_column]

            # Run inference
            response = audio_model.infer(audio_path, prompt)

            # Parse response
            response_json = parse_emotion_response(response)
            new_row.update(response_json)

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            new_row["acoustic_emotion"] = ""
            new_row["semantic_sentiment"] = ""

        # Remove heavy audio data from output
        if "audio" in new_row:
            del new_row["audio"]

        rows.append(new_row)

    result_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return result_df

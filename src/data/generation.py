"""TTS-based data generation using IndexTTS.

Wraps IndexTTS v2 to generate synthetic speech samples with controlled
acoustic emotion via emotion vectors.
"""

from __future__ import annotations

import pathlib
import sys


def infer_sample(
    tts,
    speaker_audio: str,
    text: str,
    output_path: str,
    emotion_vector: list | None = None,
    verbose: bool = True,
) -> None:
    """Run inference with a pre-loaded IndexTTS2 instance."""
    if emotion_vector is None:
        emotion_vector = [0] * 8
    tts.infer(
        spk_audio_prompt=speaker_audio,
        text=text,
        output_path=output_path,
        emo_vector=emotion_vector,
        use_emo_text=False,
        use_random=False,
        verbose=verbose,
    )


def generate_sample(
    cfg_path: str,
    model_dir: str,
    speaker_audio: str,
    text: str,
    output_path: str,
    emotion_vector: list | None = None,
    use_fp16: bool = False,
    use_cuda_kernel: bool = False,
    use_deepspeed: bool = False,
) -> None:
    """Generate a single TTS sample.

    Args:
        cfg_path: Path to IndexTTS config YAML.
        model_dir: Directory with IndexTTS checkpoints.
        speaker_audio: Path to reference speaker WAV.
        text: Target sentence to synthesise.
        output_path: Where to write the output WAV.
        emotion_vector: 8-dim emotion vector (default all zeros).
        use_fp16: Whether to use half precision.
        use_cuda_kernel: Whether to use custom CUDA kernels.
        use_deepspeed: Whether to use DeepSpeed.
    """
    _third_party = pathlib.Path(__file__).resolve().parents[2] / "third_party"
    if str(_third_party) not in sys.path:
        sys.path.insert(0, str(_third_party))
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
    )
    infer_sample(tts, speaker_audio, text, output_path, emotion_vector)

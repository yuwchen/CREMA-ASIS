"""Audio data collators for fine-tuning.

Each model family needs its own collator because the processor APIs differ,
but the label-masking logic is shared in :class:`BaseAudioDataCollator`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import librosa
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base collator with shared label-masking logic
# ---------------------------------------------------------------------------

class BaseAudioDataCollator:
    """Common logic shared across model-specific collators.

    Sub-classes must implement :meth:`__call__` and may override
    :meth:`process_audio`.
    """

    ASSISTANT_TOKEN = "<|im_start|>assistant"

    def __init__(self, processor, task_prompt: str):
        self.processor = processor
        self.task_prompt = task_prompt
        self.max_length = processor.feature_extractor.n_samples
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.assistant_start_tokens = torch.tensor(
            processor.tokenizer.encode(self.ASSISTANT_TOKEN, add_special_tokens=False)
        )

    def process_audio(self, audio_path: str) -> np.ndarray:
        """Load and optionally truncate audio."""
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
        if len(audio) > self.max_length:
            audio = audio[: self.max_length]
        return audio

    def mask_labels(
        self, input_ids: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Create labels that mask everything before the assistant response.

        The loss is only computed on the assistant tokens.
        """
        labels = input_ids.clone()
        for i in range(batch_size):
            row = input_ids[i]
            assistant_start_idx = -1
            for j in range(len(row) - len(self.assistant_start_tokens) + 1):
                if row[j: j + len(self.assistant_start_tokens)].equal(
                    self.assistant_start_tokens
                ):
                    assistant_start_idx = j
                    break
            if assistant_start_idx != -1:
                labels[i, : assistant_start_idx + len(self.assistant_start_tokens)] = -100
            else:
                labels[i, :] = -100
        return labels


# ---------------------------------------------------------------------------
# Qwen2-Audio collator
# ---------------------------------------------------------------------------

class QwenAudioDataCollator(BaseAudioDataCollator):
    """Collator for ``Qwen2AudioForConditionalGeneration``."""

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios, combined_texts = [], []
        for example in examples:
            try:
                audio = self.process_audio(example["audio_path"])
                audios.append(audio)

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": audio},
                            {"type": "text", "text": self.task_prompt},
                        ],
                    },
                    {"role": "assistant", "content": example["target"]},
                ]
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
                combined_texts.append(text)
            except Exception as e:
                print(f"Exception in QwenAudioDataCollator: {e}")

        inputs = self.processor(
            text=list(combined_texts),
            audios=list(audios),
            return_tensors="pt",
            padding=True,
            sampling_rate=self.sampling_rate,
        )
        labels = self.mask_labels(inputs["input_ids"], len(combined_texts))

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Audio-Flamingo3 collator
# ---------------------------------------------------------------------------

class AudioFlamingoAudioDataCollator(BaseAudioDataCollator):
    """Collator for ``AudioFlamingo3ForConditionalGeneration``.

    Audio is zero-padded to a fixed 30 s window (480 000 samples at 16 kHz)
    and the attention mask is adjusted to mark padded regions.
    """

    MAX_AUDIO_SAMPLES = 480000
    MAX_FEATURE_LEN = 750
    SYSTEM_START_TOKENS = 14  # tokens before the audio region

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios, combined_texts, valid_feature_lens = [], [], []
        for example in examples:
            try:
                audio = self.process_audio(example["audio_path"])

                # Pad to fixed window
                pad_width = self.MAX_AUDIO_SAMPLES - len(audio)
                audio_length_seconds = len(audio) / self.sampling_rate
                audio = np.pad(audio, (0, pad_width), mode="constant")

                valid_feature_len = min(
                    int(audio_length_seconds * 25), self.MAX_FEATURE_LEN
                )
                valid_feature_lens.append(valid_feature_len)
                audios.append(audio)

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.task_prompt},
                            {"type": "audio", "path": audio},
                        ],
                    },
                    {"role": "assistant", "content": example["target"]},
                ]
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
                combined_texts.append(text)
            except Exception as e:
                print(f"Exception in FlamingoAudioDataCollator: {e}")

        inputs = self.processor(
            text=list(combined_texts),
            audio=list(audios),
            return_tensors="pt",
            padding=True,
            sampling_rate=self.sampling_rate,
        )

        # Mask padded audio portions
        for i, vfl in enumerate(valid_feature_lens):
            start = self.SYSTEM_START_TOKENS + vfl
            end = self.SYSTEM_START_TOKENS + self.MAX_FEATURE_LEN
            inputs.attention_mask[i, start:end] = False

        labels = self.mask_labels(inputs["input_ids"], len(combined_texts))

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "input_features_mask": inputs.input_features_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Kimi-Audio collator
# ---------------------------------------------------------------------------

class KimiAudioDataCollator:
    """Collator for Kimi-Audio fine-tuning.

    Unlike Qwen and Flamingo, Kimi-Audio uses its own ``KimiAPromptManager``
    for tokenisation rather than a HuggingFace processor.
    """

    def __init__(self, prompt_manager, kimia_token_offset: int, output_type: str = "text"):
        self.prompt_manager = prompt_manager
        self.kimia_token_offset = kimia_token_offset
        self.output_type = output_type

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_audio_len = max(f["audio_input_ids"].shape[0] for f in features)
        max_text_len = max(f["text_input_ids"].shape[0] for f in features)
        batch_size = len(features)

        audio_input_ids = torch.zeros(batch_size, max_audio_len, dtype=torch.long)
        text_input_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        is_continuous_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.bool)
        all_audio_features = []

        for i, feat in enumerate(features):
            al = feat["audio_input_ids"].shape[0]
            tl = feat["text_input_ids"].shape[0]
            audio_input_ids[i, :al] = feat["audio_input_ids"]
            text_input_ids[i, :tl] = feat["text_input_ids"]
            is_continuous_mask[i, :al] = feat["is_continuous_mask"]
            all_audio_features.extend(feat["audio_features"])

        return {
            "input_ids": audio_input_ids,
            "text_input_ids": text_input_ids,
            "is_continuous_mask": is_continuous_mask,
            "whisper_input_feature": all_audio_features,
            "labels": text_input_ids.clone(),
        }

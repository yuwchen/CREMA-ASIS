"""Qwen2-Audio model wrapper.

Handles loading, inference, collator creation, and embedding extraction
for ``Qwen/Qwen2-Audio-7B-Instruct``.

Extraction approach:
- **LLM**: Uses ``output_hidden_states=True`` on the full model forward pass.
- **Whisper / projector**: Runs ``model.audio_tower`` separately and pushes its
  last hidden state through ``model.multi_modal_projector``, giving the same
  ``whisper`` / ``projector`` representations the other two models emit.
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from src.data.collators import QwenAudioDataCollator
from src.models.base import AudioModel


class Qwen2AudioModel(AudioModel):
    """Wrapper for Qwen2-Audio-7B-Instruct."""

    MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, device: str = "auto", lora_path: Optional[str] = None) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, device_map=device
        )
        if lora_path:
            print(f"Loading LoRA weights from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("LoRA weights loaded successfully!")
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, audio_path: str, prompt: str) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        audio, _ = librosa.load(audio_path, sr=sr)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Collect audio arrays from conversation
        audios = [audio]

        inputs = self.processor(
            text=text,
            audios=audios,
            return_tensors="pt",
            sampling_rate=sr,
            padding=True,
        ).to(self.model.device)

        generate_ids = self.model.generate(**inputs, max_length=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    # ------------------------------------------------------------------
    # Collator / training helpers
    # ------------------------------------------------------------------

    def get_collator(self, task_prompt: str):
        return QwenAudioDataCollator(processor=self.processor, task_prompt=task_prompt)

    def get_lora_config(self, model_config: dict) -> LoraConfig:
        lora_cfg = model_config["training"]["lora"]
        return LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            use_rslora=lora_cfg.get("use_rslora", True),
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )

    def get_trainable_model(self):
        return self.model

    # ------------------------------------------------------------------
    # Embedding extraction helpers
    # ------------------------------------------------------------------

    def get_audio_token_range(self, inputs, hidden_state):
        """Get the index range of audio tokens in the hidden states."""
        audio_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        audio_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        audio_bos_pos = (inputs["input_ids"] == audio_start_id).nonzero(as_tuple=True)[1]
        audio_eos_pos = (inputs["input_ids"] == audio_end_id).nonzero(as_tuple=True)[1]

        if len(audio_bos_pos) == 0 or len(audio_eos_pos) == 0:
            return None, None

        audio_bos_idx = audio_bos_pos[0].item()
        num_audio_embeddings = hidden_state.shape[1] - inputs["input_ids"].shape[1] + 1
        audio_start = audio_bos_idx + 1
        audio_end = audio_start + num_audio_embeddings
        return audio_start, audio_end

    def extract_llm_embeddings(
        self,
        audio_path: str,
        selected_layers: list[int],
        prompt: str,
        role_prompt: str = "You are a helpful assistant.",
        use_float16: bool = True,
        device: str = "cuda:0",
    ) -> dict:
        """Extract LLM hidden-state embeddings using ``output_hidden_states``.

        Returns:
            Dict ``{layer_idx: {"mean": tensor, "last": tensor}}``.
        """
        sr = self.processor.feature_extractor.sampling_rate
        conversation = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # Assuming we only use a single audio input (single turn chat)
        audio_data, _ = librosa.load(audio_path, sr=sr)
        inputs = self.processor(
            text=text, audios=[audio_data], return_tensors="pt",
            padding=True, sampling_rate=sr,
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states
            total_layers = len(all_hidden_states)

            resolved_layers = [
                idx if idx >= 0 else total_layers + idx for idx in selected_layers
            ]

            file_embeddings = {}
            for layer_idx in resolved_layers:
                hs = all_hidden_states[layer_idx][0].cpu().numpy()
                if use_float16:
                    hs = hs.astype(np.float16)

                pooled = {
                    "last": torch.from_numpy(hs[-1]),
                    "mean": torch.from_numpy(hs.mean(axis=0)),
                }
                file_embeddings[layer_idx] = pooled

        return file_embeddings

    def _get_audio_modules(self):
        """Return ``(audio_tower, multi_modal_projector)``, unwrapping PEFT."""
        model = self.model
        if hasattr(model, "base_model"):
            base = model.base_model
            if not hasattr(base, "audio_tower"):
                base = base.model
        else:
            base = model
        return base.audio_tower, base.multi_modal_projector

    def extract_encoder_embeddings(
        self,
        audio_path: str,
        selected_layers: list = None,
        prompt: str = "",
        role_prompt: str = "You are a helpful assistant.",
        use_float16: bool = True,
        device: str = "cuda:0",
    ) -> dict:
        """Extract Whisper encoder and multi-modal projector representations.

        The audio tower is run once; its last hidden state is the ``whisper``
        representation and is then pushed through ``multi_modal_projector`` to
        give the ``projector`` representation (1280-dim -> 4096-dim).  This
        mirrors what the Kimi-Audio and Audio-Flamingo3 extractors emit, so
        probing results are directly comparable across the three models.

        Args:
            audio_path: Path to a WAV file.
            selected_layers: Unused; the layer selection for this component is
                fixed to the last encoder layer.  Accepted so the call
                signature matches the other extractors.
            prompt: Task prompt text.
            role_prompt: System prompt.
            use_float16: Store embeddings as FP16.
            device: Torch device string.

        Returns:
            Dict ``{"whisper": {"mean": tensor, "last": tensor},
            "projector": {"mean": tensor, "last": tensor}}``.
        """
        sr = self.processor.feature_extractor.sampling_rate
        conversation = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio_data, _ = librosa.load(audio_path, sr=sr)
        inputs = self.processor(
            text=text, audios=[audio_data], return_tensors="pt",
            padding=True, sampling_rate=sr,
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        audio_encoder, projector = self._get_audio_modules()
        audio_encoder.eval()
        projector.eval()

        with torch.no_grad():
            encoder_outputs = audio_encoder(inputs["input_features"])
            if hasattr(encoder_outputs, "last_hidden_state"):
                encoder_hidden = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, tuple):
                encoder_hidden = encoder_outputs[0]
            else:
                encoder_hidden = encoder_outputs

            # projector: (batch, seq, 1280) -> (batch, seq, 4096)
            projected = projector(encoder_hidden)

            file_embeddings = {}
            for name, states in (("whisper", encoder_hidden), ("projector", projected)):
                hs = states[0].float().cpu().numpy()
                if use_float16:
                    hs = hs.astype(np.float16)
                file_embeddings[name] = {
                    "last": torch.from_numpy(hs[-1]),
                    "mean": torch.from_numpy(hs.mean(axis=0)),
                }

        return file_embeddings

    # Backwards-compatible alias for the pre-projector entry point.
    extract_whisper_embeddings = extract_encoder_embeddings

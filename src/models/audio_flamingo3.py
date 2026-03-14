"""Audio-Flamingo3 model wrapper.

Handles loading, inference, collator creation, and hook-based embedding
extraction for ``nvidia/audio-flamingo-3-hf``.

Extraction approach: forward hooks on selected LLM layers, the audio tower
(Whisper last layer), and the multi-modal projector.

Training note: both the LLM and Whisper encoder are trained (LoRA on
``all-linear`` covers both).
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.data.collators import AudioFlamingoAudioDataCollator
from src.models.base import AudioModel


class AudioFlamingo3Model(AudioModel):
    """Wrapper for Audio-Flamingo3."""

    MODEL_NAME = "nvidia/audio-flamingo-3-hf"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, device: str = "auto", lora_path: Optional[str] = None) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
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
        sr = 16000
        audio, _ = librosa.load(audio_path, sr=sr)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=[audio], sampling_rate=sr,
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=500)
        decoded = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded[0]

    # ------------------------------------------------------------------
    # Collator / training helpers
    # ------------------------------------------------------------------

    def get_collator(self, task_prompt: str):
        return AudioFlamingoAudioDataCollator(
            processor=self.processor, task_prompt=task_prompt
        )

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


# ---------------------------------------------------------------------------
# Audio-Flamingo3 Embedding Extractor (hook-based)
# ---------------------------------------------------------------------------

class _AudioTowerCapture:
    """Captures the last encoder layer's hidden state via a forward hook."""

    def __init__(self):
        self.last_layer_hidden_states = None
        self.hook = None

    def register(self, audio_tower):
        last_layer = audio_tower.layers[-1]
        self.hook = last_layer.register_forward_hook(self._hook)
        return self.hook

    def _hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.last_layer_hidden_states = hidden.detach().cpu().clone()

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get(self):
        return self.last_layer_hidden_states


class AudioFlamingoEmbeddingExtractor:
    """Extract layer-wise embeddings from Audio-Flamingo3 using forward hooks.

    Hooks are placed on:
    - Selected LLM transformer layers
    - The audio tower last encoder layer (Whisper)
    - The multi-modal projector
    """

    def __init__(self, model, processor, selected_layers: list[int]):
        self.model = model
        self.processor = processor
        self.selected_layers = selected_layers

        self.audio_capture = _AudioTowerCapture()
        self.captured_hidden_states = []
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured_hidden_states.append(output[0].detach().cpu().clone())
            else:
                self.captured_hidden_states.append(output.detach().cpu().clone())

        # LLM layers
        for i, layer in enumerate(self.model.language_model.model.layers):
            if i in self.selected_layers:
                self.hooks.append(layer.register_forward_hook(hook_fn))

        # Audio tower (Whisper last layer)
        self.hooks.append(self.audio_capture.register(self.model.audio_tower))

        # Multi-modal projector
        self.hooks.append(
            self.model.multi_modal_projector.register_forward_hook(hook_fn)
        )

    def unregister_hooks(self):
        self.audio_capture.remove()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_file(
        self,
        audio_path: str,
        prompt: str,
        role_prompt: str = "You are a helpful assistant.",
        use_float16: bool = True,
    ) -> dict:
        """Extract pooled embeddings for a single audio file.

        Returns:
            Dict ``{layer_name: {"mean": tensor, "last": tensor}}``.
        """
        self.captured_hidden_states = []

        sr = 16000
        audio, _ = librosa.load(audio_path, sr=sr)

        conversation = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self.model.device)

        with torch.inference_mode():
            self.model(**inputs, output_hidden_states=True)

        # Insert whisper output at position 0
        audio_hs = self.audio_capture.get()
        self.captured_hidden_states.insert(0, audio_hs)

        layer_labels = ["whisper", "projector"] + self.selected_layers

        file_embeddings = {}
        for hs, name in zip(self.captured_hidden_states, layer_labels):
            h = hs[0]  # remove batch dim
            if h.dtype == torch.bfloat16:
                h = h.float()
            h_np = h.numpy()
            if use_float16:
                h_np = h_np.astype(np.float16)

            file_embeddings[name] = {
                "last": torch.from_numpy(h_np[-1]),
                "mean": torch.from_numpy(h_np.mean(axis=0)),
            }

        return file_embeddings

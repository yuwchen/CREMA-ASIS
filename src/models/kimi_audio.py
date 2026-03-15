"""Kimi-Audio model wrapper.

Handles loading, inference, collator creation, and hook-based embedding
extraction for ``moonshotai/Kimi-Audio-7B-Instruct``.

Extraction approach: forward hooks on selected LLM layers, the Whisper
encoder, the VQ adaptor, and the MIMO layer.

Training note: only the ALM (language model) is trained with LoRA.
The Whisper encoder is frozen.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel

from src.data.collators import KimiAudioDataCollator
from src.models.base import AudioModel


# ---------------------------------------------------------------------------
# PEFT compatibility helper (must be monkey-patched onto the ALM)
# ---------------------------------------------------------------------------

def _prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None,
    inputs_embeds=None, **kwargs,
):
    """``prepare_inputs_for_generation`` required by PEFT for Kimi-Audio."""
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }
    if inputs_embeds is not None and past_key_values is None:
        model_inputs["inputs_embeds"] = inputs_embeds
        model_inputs["input_ids"] = None
    return model_inputs


class KimiAudioModel(AudioModel):
    """Wrapper for Kimi-Audio-7B-Instruct."""

    MODEL_NAME = "moonshotai/Kimi-Audio-7B-Instruct"

    DEFAULT_SAMPLING_PARAMS = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        self.model = None  # KimiAudio instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, device: str = "auto", lora_path: Optional[str] = None) -> None:
        import sys, pathlib
        _third_party = pathlib.Path(__file__).resolve().parents[2] / "third_party"
        if str(_third_party) not in sys.path:
            sys.path.insert(0, str(_third_party))
        from kimia_infer.api.kimia import KimiAudio

        # KimiAudio.__init__ hardcodes .to(torch.cuda.current_device()) for both
        # the ALM and the audio tokenizer. With DeepSpeed multi-GPU, all ranks
        # default to device 0, putting multiple full models on one GPU.
        # Setting the device to LOCAL_RANK first distributes each rank to its GPU.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        self.model = KimiAudio(model_path=self.model_name, load_detokenizer=True)

        if lora_path:
            print(f"Loading LoRA weights from: {lora_path}")
            self.model.alm.prepare_inputs_for_generation = (
                _prepare_inputs_for_generation.__get__(self.model.alm)
            )
            self.model.alm = PeftModel.from_pretrained(self.model.alm, lora_path)
            print("LoRA weights loaded successfully!")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, audio_path: str, prompt: str) -> str:
        """Run inference.  Audio must be a file path (Kimi API requirement)."""
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        _, text_output = self.model.generate(
            messages,
            **self.DEFAULT_SAMPLING_PARAMS,
            max_new_tokens=70,
            output_type="text",
        )
        return text_output

    # ------------------------------------------------------------------
    # Collator / training helpers
    # ------------------------------------------------------------------

    def get_collator(self, task_prompt: str):
        """Return KimiAudioDataCollator (requires prompt_manager)."""
        return KimiAudioDataCollator(
            prompt_manager=self.model.prompt_manager,
            kimia_token_offset=self.model.kimia_token_offset,
            task_prompt=task_prompt,
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
        """Only the ALM is trained."""
        return self.model.alm

    def prepare_for_training(self):
        """Monkey-patch and prepare the ALM for PEFT training."""
        self.model.alm.gradient_checkpointing_enable()
        self.model.alm.prepare_inputs_for_generation = (
            _prepare_inputs_for_generation.__get__(self.model.alm)
        )


# ---------------------------------------------------------------------------
# Kimi-Audio Embedding Extractor (hook-based)
# ---------------------------------------------------------------------------

class KimiAudioEmbeddingExtractor:
    """Extract layer-wise embeddings from Kimi-Audio using forward hooks.

    Hooks are placed on:
    - Selected LLM (ALM) transformer layers
    - The Whisper speech encoder (captures last hidden state)
    - The VQ adaptor (projector)
    - The MIMO layer
    """

    def __init__(self, model_path: str, selected_layers: list[int]):
        from kimia_infer.api.kimia import KimiAudio

        self.kimi = KimiAudio(model_path=model_path, load_detokenizer=False)
        self.alm = self.kimi.alm
        self.prompt_manager = self.kimi.prompt_manager
        self.selected_layers = selected_layers

        self.captured_hidden_states = []
        self.layer_names = []
        self.hooks = []

        self.alm.eval()
        self.prompt_manager.whisper_model.speech_encoder.eval()
        self._register_hooks()

    # -- Hook management ---------------------------------------------------

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured_hidden_states.append(output[0].detach().cpu().clone())
            else:
                self.captured_hidden_states.append(output.detach().cpu().clone())
            self.layer_names.append(module)

        def whisper_hook_fn(module, input, output):
            self.captured_hidden_states.append(
                output.hidden_states[-1].detach().cpu().clone()
            )
            self.layer_names.append(module)

        # Selected ALM layers
        for i, layer in enumerate(self.alm.model.layers):
            if i in self.selected_layers:
                self.hooks.append(layer.register_forward_hook(hook_fn))

        # MIMO layer
        self.hooks.append(
            self.alm.model.mimo_layers[0].register_forward_hook(hook_fn)
        )

        # Whisper encoder
        self.hooks.append(
            self.prompt_manager.whisper_model.speech_encoder.register_forward_hook(
                whisper_hook_fn
            )
        )

        # VQ adaptor (projector)
        self.hooks.append(
            self.kimi.alm.model.vq_adaptor.register_forward_hook(hook_fn)
        )

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # -- Extraction --------------------------------------------------------

    def extract_hidden_states(self, messages: list[dict]) -> dict:
        """Run a forward pass and return captured hidden states.

        Args:
            messages: Chat-format messages (same as ``KimiAudio.generate``).

        Returns:
            Dict with ``hidden_states`` list and ``layer_names`` list.
            Order: [whisper, projector, *selected_llm_layers, MIMO].
        """
        self.captured_hidden_states = []
        self.layer_names = []

        history = self.prompt_manager.get_prompt(messages, output_type="text")
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature

        device = torch.cuda.current_device()
        audio_input_ids = audio_input_ids.to(device)
        text_input_ids = text_input_ids.to(device)
        is_continuous_mask = is_continuous_mask.to(device)
        audio_features = [f.to(device) for f in audio_features]

        position_ids = torch.arange(
            0, audio_input_ids.shape[1], device=device
        ).unsqueeze(0).long()

        with torch.inference_mode():
            self.alm.forward(
                input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=audio_features,
                is_continuous_mask=is_continuous_mask,
                position_ids=position_ids,
                past_key_values=None,
                return_dict=False,
                output_hidden_states=True,
            )

        # Fix projector shape (comes transposed)
        if len(self.captured_hidden_states) > 1:
            self.captured_hidden_states[1] = self.captured_hidden_states[1].transpose(1, 0)

        layer_labels = ["whisper", "projector"] + self.selected_layers + ["MIMO"]
        return {
            "hidden_states": self.captured_hidden_states,
            "layer_names": layer_labels,
        }

    def extract_file(
        self,
        audio_path: str,
        prompt: str,
        use_float16: bool = True,
    ) -> dict:
        """Extract pooled embeddings for a single audio file.

        Returns:
            Dict ``{layer_name: {"mean": tensor, "last": tensor}}``.
        """
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        result = self.extract_hidden_states(messages)

        file_embeddings = {}
        for idx, (hs, name) in enumerate(
            zip(result["hidden_states"], result["layer_names"])
        ):
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

    def load_lora(self, lora_path: str):
        """Load LoRA weights onto the ALM for extraction."""
        self.kimi.alm.prepare_inputs_for_generation = (
            _prepare_inputs_for_generation.__get__(self.kimi.alm)
        )
        self.kimi.alm = PeftModel.from_pretrained(self.kimi.alm, lora_path)
        self.alm = self.kimi.alm
        print(f"LoRA loaded from {lora_path}")

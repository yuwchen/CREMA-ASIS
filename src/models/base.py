"""Abstract base classes for audio model wrappers.

Each supported model family implements :class:`AudioModel` which provides
a uniform interface for loading, inference, collator creation, and
embedding extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class AudioModel(ABC):
    """Interface that every model wrapper must implement."""

    @abstractmethod
    def load(
        self,
        device: str = "auto",
        lora_path: Optional[str] = None,
    ) -> None:
        """Load the model (and optionally LoRA weights) onto *device*."""
        ...

    @abstractmethod
    def infer(self, audio_path: str, prompt: str) -> str:
        """Run inference on a single audio file.

        Args:
            audio_path: Path to a WAV file.
            prompt: Task prompt text.

        Returns:
            Raw text response from the model.
        """
        ...

    @abstractmethod
    def get_collator(self, task_prompt: str):
        """Return a data collator suitable for fine-tuning this model."""
        ...

    @abstractmethod
    def get_lora_config(self, model_config: dict):
        """Return a ``LoraConfig`` built from *model_config*."""
        ...

    @abstractmethod
    def get_trainable_model(self):
        """Return the model (or sub-model) to which LoRA should be applied."""
        ...

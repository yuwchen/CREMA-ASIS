"""Model registry — maps short names to model wrapper classes."""

from __future__ import annotations

from src.models.base import AudioModel


MODEL_REGISTRY: dict[str, type[AudioModel]] = {
    "qwen2-audio": None,
    "kimi-audio": None,
    "flamingo3": None,
}


def get_model(name: str, **kwargs) -> AudioModel:
    """Instantiate a model wrapper by short name.

    Args:
        name: One of ``"qwen2-audio"``, ``"kimi-audio"``, ``"flamingo3"``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        An :class:`~src.models.base.AudioModel` instance (not yet loaded).

    Raises:
        KeyError: If *name* is not in the registry.
    """

    
    
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # transformers version incompatability - won't always be able to import everything
    if name == 'qwen2-audio':
        from src.models.qwen2_audio import Qwen2AudioModel
        MODEL_REGISTRY[name] = Qwen2AudioModel
    if name == 'kimi-audio':
        from src.models.kimi_audio import KimiAudioModel
        MODEL_REGISTRY[name] = KimiAudioModel
    if name == 'flamingo3':
        from src.models.flamingo3 import Flamingo3Model
        MODEL_REGISTRY[name] = Flamingo3Model
        
    
    return MODEL_REGISTRY[name](**kwargs)

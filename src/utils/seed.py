"""Unified random seed setter for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

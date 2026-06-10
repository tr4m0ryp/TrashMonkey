"""Determinism helpers. Keep minimal.

Config access lives in yolo_waste_sorter.utils.config (typed loader).
"""

import os
import random


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
<<<<<<< HEAD
=======


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"config root must be a mapping, got {type(config).__name__}: {path}")
    return config
>>>>>>> agent/agent-a8e06b4e3d57af60c

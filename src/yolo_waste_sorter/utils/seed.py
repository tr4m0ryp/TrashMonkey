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
        config: dict[str, Any] = yaml.safe_load(f)
    return config
>>>>>>> agent/agent-aaf5ec144e4997d51

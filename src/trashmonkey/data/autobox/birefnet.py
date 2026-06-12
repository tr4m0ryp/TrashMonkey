"""Fallback backend: rembg BiRefNet alpha mask (lazy import).

rembg ships the ``birefnet-general`` session in its base package (verified at
rembg 2.0.59+); the ``[cpu]``/``[gpu]`` extra only selects the onnxruntime
build. rembg uses every available onnxruntime provider, so CUDA is picked up
automatically when onnxruntime-gpu is installed, else CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from trashmonkey.data.autobox.types import INSTALL_HINT, MaskFn

BIREFNET_SESSION = "birefnet-general"


def build_birefnet_backend(model_name: str = BIREFNET_SESSION) -> MaskFn:
    """Build a mask closure over a loaded rembg BiRefNet session.

    Verified against rembg 2.0.76: ``new_session(name)`` and
    ``remove(pil_image, session=..., only_mask=True) -> PIL 'L' image``.
    """
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise ImportError(INSTALL_HINT.format(backend="birefnet", package="rembg")) from exc

    session = new_session(model_name)

    def predict_mask(image_path: Path) -> npt.NDArray[np.uint8]:
        with Image.open(image_path) as img:
            mask = remove(img.convert("RGB"), session=session, only_mask=True)
        return np.asarray(mask, dtype=np.uint8)

    return predict_mask

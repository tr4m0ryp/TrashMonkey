"""Primary backend: Grounding DINO via HuggingFace transformers (lazy import).

The detector is used for LOCALIZATION ONLY: a single caption per material class
proposes boxes, and the class ID is supplied by the source mapping. We use the
transformers-native Grounding DINO (``IDEA-Research/grounding-dino-tiny``,
``AutoModelForZeroShotObjectDetection``) rather than the third-party
``autodistill``/``groundingdino`` stack: the transformers implementation tracks
the installed transformers version, so it does not break against new BERT
internals (the ``groundingdino`` fork does). CUDA is used when available, else
CPU. Weights download from the HF Hub on first use (public model, no token).
"""

from __future__ import annotations

from pathlib import Path

from trashmonkey.data.autobox.types import (
    DEFAULT_TEXT_THRESHOLD,
    INSTALL_HINT,
    MIN_BOX_CONFIDENCE,
    Detection,
    DinoPredictFn,
)

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"


def _normalize_prompt(prompt: str) -> str:
    """Grounding DINO wants a lowercase caption terminated by a period."""
    text = " ".join(prompt.lower().split())
    return text if text.endswith(".") else text + " ."


def build_dino_backend(
    prompt: str,
    *,
    box_threshold: float = MIN_BOX_CONFIDENCE,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> DinoPredictFn:
    """Build a predictor closure over a loaded transformers Grounding DINO model.

    The model + processor load once and are reused across every image. ``predict``
    returns pixel-space ``Detection``s (xyxy, confidence) at or above the
    thresholds; the caller filters/sorts and assigns the class ID.
    """
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise ImportError(INSTALL_HINT.format(backend="dino", package="transformers")) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL)
    model = model.to(device).eval()
    text = _normalize_prompt(prompt)

    def _post_process(outputs: object, input_ids: object, target_sizes: object) -> dict[str, object]:
        # `box_threshold` was renamed to `threshold` in transformers >= 4.51;
        # try the current name first, fall back to the legacy one.
        fn = processor.post_process_grounded_object_detection
        try:
            return fn(
                outputs,
                input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            return fn(
                outputs,
                input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]

    from PIL import Image

    def predict(image_path: Path) -> list[Detection]:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            width, height = image.size
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = _post_process(outputs, inputs["input_ids"], [(height, width)])
        boxes = result["boxes"].tolist()  # type: ignore[attr-defined]
        scores = result["scores"].tolist()  # type: ignore[attr-defined]
        return [
            Detection(xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), confidence=float(s))
            for b, s in zip(boxes, scores, strict=True)
        ]

    return predict

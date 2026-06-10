"""Primary backend: Grounding DINO via autodistill-grounding-dino (lazy import).

The detector is used for LOCALIZATION ONLY: a single caption per material
class proposes boxes, and the class ID is supplied by the source mapping.
autodistill-grounding-dino selects CUDA when available, else CPU (module-level
``torch.device`` in the package).
"""

from __future__ import annotations

from pathlib import Path

from yolo_waste_sorter.data.autobox.types import (
    DEFAULT_TEXT_THRESHOLD,
    INSTALL_HINT,
    MIN_BOX_CONFIDENCE,
    Detection,
    DinoPredictFn,
)


def build_dino_backend(
    prompt: str,
    *,
    box_threshold: float = MIN_BOX_CONFIDENCE,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> DinoPredictFn:
    """Build a predictor closure over a loaded Grounding DINO (Swin-T) model.

    Verified against autodistill-grounding-dino 0.1.4: ``GroundingDINO(ontology=
    CaptionOntology({caption: label}), box_threshold=..., text_threshold=...)``
    with ``predict(path) -> supervision.Detections`` (fields .xyxy, .confidence).
    """
    try:
        from autodistill.detection import CaptionOntology
        from autodistill_grounding_dino import GroundingDINO
    except ImportError as exc:
        raise ImportError(
            INSTALL_HINT.format(backend="dino", package="autodistill-grounding-dino")
        ) from exc

    model = GroundingDINO(
        ontology=CaptionOntology({prompt: prompt}),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    def predict(image_path: Path) -> list[Detection]:
        detections = model.predict(str(image_path))
        if detections.confidence is None or len(detections.xyxy) == 0:
            return []
        return [
            Detection(
                xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                confidence=float(c),
            )
            for b, c in zip(detections.xyxy, detections.confidence, strict=True)
        ]

    return predict

"""ESP32-CAM (OV2640) degradation transforms -- the shared camera model.

One module, one camera model: training augmentation (T5), TEST-2 evaluation
generation (T6), and threshold tuning (T9) all import from here so every
consumer sees the same simulated deployment camera.

Public surface:
- ``build_train_stack(cfg)`` -- Albumentations image-only transforms for
  train-time augmentation, parameterized from ``cfg.augment.esp32_stack``.
- ``degrade_image(img, severity, seed)`` -- deterministic severity-graded
  (1..5) degradation mimicking the deployment path; byte-identical for
  identical ``(img, severity, seed)``.
"""

from yolo_waste_sorter.utils.degrade.severity import degrade_image
from yolo_waste_sorter.utils.degrade.train_stack import build_train_stack

__all__ = ["build_train_stack", "degrade_image"]

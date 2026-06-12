# yolo-waste-sorter

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Ultralytics YOLO11](https://img.shields.io/badge/ultralytics-%E2%89%A58.3.226-111f68.svg)](https://docs.ultralytics.com/models/yolo11/)
[![TensorRT FP16](https://img.shields.io/badge/deploy-TensorRT%20FP16%20%C2%B7%20Jetson%20Orin%20Nano-76b900.svg)](src/yolo_waste_sorter/deploy/)
[![Checks](https://img.shields.io/badge/checks-pytest%20%C2%B7%20mypy--strict%20%C2%B7%20ruff-brightgreen.svg)](Makefile)

**YOLO11n waste detector for a physical sorting machine.** Six material
classes -- plastic, paper, cardboard, metal, glass, organic -- detected as
single objects on a white conveyor background, trained entirely on remapped
public datasets and deployed as a TensorRT FP16 engine on an NVIDIA Jetson
Orin Nano watching three ESP32-CAM streams. There is no trained catch-all
class: items are binned by a **multi-frame consensus rule**, and anything that
fails it rides off into a "rest" bin.

> Course project with a live demo target. Every reported number ties to a run
> config and seeds are fixed.

## How it works

1. **Data pipeline** (`make repro`) -- fetch five public sources (TrashNet,
   Drinking Waste, Garbage Classification 3, household-waste; RealWaste is
   held out entirely as the unseen test set), remap labels to the six classes,
   auto-box classification-only images with Grounding DINO (BiRefNet and
   center-box fallbacks + a QA gate), pHash-dedup across sources, cap-balance,
   and emit one YOLO dataset. Dropped categories become the open-set probe
   used for threshold tuning -- never training data.
2. **Training** -- pinned AdamW recipe on COCO-pretrained `yolo11n.pt`, with an
   ESP32-CAM degradation stack (JPEG blocking, sensor noise, motion blur,
   white-balance drift) injected at train time so cheap-camera artifacts are
   learned, not feared. Runs on Colab via `notebooks/manager.ipynb`.
3. **Evaluation** -- three tiers: stratified validation, leave-one-source-out
   test, and degraded copies of that test set as the demo-day proxy.
4. **Thresholds** -- a tuner sweeps the consensus rule (per-frame tau, vote
   count, high-water mark) against wrong-bin vs rest-rate and emits
   `thresholds.yaml`, the deployment artifact next to the engine.
5. **Jetson runtime** -- grab-latest MJPEG readers, round-robin inference
   (~5 ms/frame FP16), per-object vote aggregation, and a `(class, confidence,
   timestamp)` handoff to the control logic.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]"

make test            # unit + integration suite
FAKE_MODEL=1 make smoke   # full pipeline -> train -> eval -> thresholds on synthetic data
make repro           # real data pipeline (downloads public datasets)
```

GPU training runs through `notebooks/manager.ipynb` (Colab-ready, run-all
safe). Jetson deployment scripts live in `src/yolo_waste_sorter/deploy/` --
the TensorRT engine must be exported on the device itself.

## Layout

Hyperparameters and the label map live in `configs/`; all logic in
`src/yolo_waste_sorter/` (data pipeline, training, evaluation, thresholding,
deploy); notebooks only orchestrate. The design rationale with full citations
is in `research/yolo11-waste-detection-finetune.md`.

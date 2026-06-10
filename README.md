# yolo-waste-sorter

Fine-tuning YOLO11n for six-class waste detection on embedded hardware.

Detects plastic, paper, cardboard, metal, glass, and organic waste on a white
background, built entirely from remapped public datasets (TACO, TrashNet,
Kaggle garbage sets, Drinking Waste). Deployed as a TensorRT FP16 engine on an
NVIDIA Jetson Orin Nano fed by ESP32-CAM streams. Objects below the confidence
threshold fall through to a "rest" bin by control logic -- no catch-all class
is trained.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]"
```

## Reproduce

`make repro` runs the data pipeline (download, remap to the six-class label
map, merge, balance), training, and export. Seeds and hyperparameters live in
`configs/config.yaml`. The technical report builds with `make paper`.

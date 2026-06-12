# yolo-waste-sorter

Trains a small, fast vision model (YOLO11n) to recognize six kinds of waste
in a photo: **plastic, paper, cardboard, metal, glass and organic**. When it
isn't sure, it says **"rest"** instead of guessing -- better no answer than a
wrong one.

Everything is built from **free public datasets** -- no private photos --
and the finished model can run on cheap hardware. The expected input is one
item at a time on a white background.

## What's inside

- A data pipeline that downloads 5 public datasets and merges them into one
  training set (relabeling, drawing missing boxes, removing duplicates)
- A training recipe that runs on Google Colab
- An evaluation on data the model has never seen
- A tuner that picks the "rest" decision thresholds
- An export step that packages the model for any device (ONNX by default,
  TensorRT for NVIDIA boards)

This repository stops at the finished model files. Connecting them to
cameras or machines is up to whatever system uses the model.

## Try it

```bash
git clone <repo-url> && cd yolo-waste-sorter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

FAKE_MODEL=1 make smoke   # dry-runs the whole pipeline offline, ~1 minute
make test                 # run the test suite
```

The smoke run walks through every stage (data, training, evaluation,
thresholds) on tiny fake data, so you can check everything works without
downloading anything.

## Train it for real

**1. Build the dataset** (needs free Kaggle credentials in
`~/.kaggle/kaggle.json`):

```bash
pip install -e ".[boxing]"
make repro
```

This downloads the five datasets, relabels everything into the six classes,
draws bounding boxes where the source only has labels, removes duplicate
images, balances the classes and produces one ready-to-train dataset. If the
automatic boxes look suspicious, the pipeline stops and asks you to review
them first.

**2. Train.** Open `notebooks/manager.ipynb` in Google Colab and choose
"Runtime > Run all" -- it handles everything, including resuming after a
disconnect. Or locally:

```bash
python -m yolo_waste_sorter.models.train --smoke   # quick sanity check
python -m yolo_waste_sorter.models.train           # full training run
```

**3. Evaluate and tune.** The model is scored three ways: on a normal
validation split, on a full dataset it never trained on (RealWaste), and on
a deliberately degraded copy that mimics cheap cameras. Then a sweep picks
the "rest" thresholds:

```bash
python -m yolo_waste_sorter.models.evaluate
python -m yolo_waste_sorter.models.thresholds
```

**4. Export.** Package the trained model for wherever it needs to run:

```bash
python -m yolo_waste_sorter.deploy.export --weights models/best.pt                  # ONNX, runs almost anywhere
python -m yolo_waste_sorter.deploy.export --weights models/best.pt --format engine  # TensorRT, run this on the NVIDIA device itself
```

Every export is test-fired once before it's trusted. The exported model plus
the generated `thresholds.yaml` are the two files a downstream system needs.

## How the "rest" decision works

"Rest" is not something the model was taught -- it's a simple rule applied
to the model's output. Watching one item over several video frames, the item
only gets a material class if the frames **agree often enough** (at least 3
votes), the winning class has a **clear majority**, and at least one frame
was **genuinely confident** (score 0.60+). Anything weaker becomes "rest".
This works much better against unfamiliar objects than a single confidence
check, because a fluke has to repeat itself to win.

## The datasets

| Source | License | Images | Used for |
|---|---|---|---|
| [TrashNet](https://github.com/garythung/trashnet) | MIT | 2,527 | training |
| [Drinking Waste](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification) | CC0-1.0 | 4,828 | training |
| [Garbage Detection](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection) | CC BY 4.0 | 10,464 | training |
| [Recyclable & Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) | MIT | 15,000 | training |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) | CC BY 4.0 | 4,752 | testing only -- never trained on |

How every source label maps onto the six classes is spelled out in
`configs/datasets.yaml`.

## Good to know

- **One item, plain background.** The model is not built for cluttered
  scenes with many objects.
- **Reproducible by design.** Fixed random seed (42) everywhere, pinned
  dependencies, checksummed downloads; identical inputs give identical
  outputs. Every training run logs its config and results to
  `experiments/runs.jsonl`.
- **No results yet.** The full pipeline is built and tested end to end, but
  the first complete GPU training run is still on the roadmap -- no accuracy
  numbers are claimed until then.
- **License:** AGPL-3.0-or-later (see `LICENSE`), inherited from the
  Ultralytics ecosystem. Dataset licenses are listed above.

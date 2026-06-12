# TrashMonkey

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tr4m0ryp/TrashMonkey/blob/main/notebooks/manager.ipynb)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Model](https://img.shields.io/badge/model-YOLO11n-orange.svg)
![Seed](https://img.shields.io/badge/seed-42-green.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)

Trains a small, fast vision model (YOLO11n) to recognize six kinds of waste
in a photo.

| plastic | paper | cardboard | metal | glass | organic |
|:-:|:-:|:-:|:-:|:-:|:-:|

When it isn't sure, it says **"rest"** instead of guessing -- better no
answer than a wrong one.

Everything is built from **free public datasets** -- no private photos --
and the finished model can run on cheap hardware. Training images show items
on a white background, but because the model fine-tunes a general-purpose,
COCO-pretrained YOLO11, it keeps broad detection ability: busier or less
clean backgrounds still work, with the best accuracy on plain ones.

## What's inside

| Stage | What it does |
|---|---|
| **Data pipeline** | Downloads 5 public datasets and merges them into one training set: relabeling, drawing missing boxes, removing duplicates |
| **Training** | Fine-tunes YOLO11n on the merged set; runs on Google Colab |
| **Evaluation** | Scores the model on data it has never seen |
| **Threshold tuner** | Picks the "rest" decision thresholds automatically |
| **Export** | Packages the model for any device (ONNX by default, TensorRT for NVIDIA boards) |

## Try it

Open [`notebooks/manager.ipynb`](notebooks/manager.ipynb) in Google Colab
and choose **"Runtime > Run all"** -- the notebook is the manager and drives
the whole project: setup checks, data, training, evaluation and plots, with
zero logic of its own. It connects your Google Drive so checkpoints and
results survive disconnects, runs an automatic smoke test before training,
and resumes from the last checkpoint if the session dies. Prefer a terminal?
`FAKE_MODEL=1 make smoke` dry-runs the entire pipeline offline in about a
minute.

<details>
<summary>Terminal commands (without the notebook)</summary>

```bash
git clone https://github.com/tr4m0ryp/TrashMonkey.git && cd TrashMonkey
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

FAKE_MODEL=1 make smoke                            # offline dry run
make test                                          # test suite

pip install -e ".[boxing]" && make repro           # build the real dataset (Kaggle creds)
python -m yolo_waste_sorter.models.train           # train
python -m yolo_waste_sorter.models.evaluate        # three-tier evaluation
python -m yolo_waste_sorter.models.thresholds      # tune the "rest" thresholds
python -m yolo_waste_sorter.deploy.export --weights models/best.pt   # export (ONNX)
```
</details>

## YOLO and the data

**YOLO** ("You Only Look Once") is a family of object detectors that find
and classify everything in an image in a single pass, which is what makes
them fast enough for live video on small hardware. **YOLO11n** is the
smallest member of the current Ultralytics generation; we start from weights
pretrained on the general-purpose COCO dataset and fine-tune them for waste.

The training set is merged from five public sources. Their original labels
(30+ categories, different naming schemes, some without bounding boxes) are
remapped onto the six classes, missing boxes are drawn automatically,
near-duplicate images are removed across sources, and classes are balanced.
One source is never trained on at all, so it can serve as an honest test:

| Source | License | Images | Used for |
|---|---|---|---|
| [TrashNet](https://github.com/garythung/trashnet) | MIT | 2,527 | training |
| [Drinking Waste](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification) | CC0-1.0 | 4,828 | training |
| [Garbage Detection](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection) | CC BY 4.0 | 10,464 | training |
| [Recyclable & Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) | MIT | 15,000 | training |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) | CC BY 4.0 | 4,752 | **testing only -- never trained on** |

The full label mapping lives in `configs/datasets.yaml`.

## How the fine-tuning works

We take the COCO-pretrained `yolo11n.pt` and retrain **all layers** on the
merged waste dataset -- a full fine-tune rather than only swapping the
classification head, which consistently scores better on small datasets.
The recipe is pinned and deterministic: AdamW optimizer (Ultralytics'
automatic choice silently changes with dataset size, so we fix it), 100
epochs at 640 px, batch 16, seed 42 everywhere, so identical inputs give
identical models.

The augmentation is matched to the task. Waste items have no natural
orientation, so frames are flipped and rotated up to 180 degrees; hue jitter
is kept very low because color is the strongest material cue; and a
camera-degradation stack (JPEG artifacts, sensor noise, motion blur,
white-balance drift, downscaling) is injected at train time so images from
cheap cameras don't surprise the model later.

**"Rest" is not a trained class.** A rule applied after detection watches an
object over several frames and only assigns a material class if at least 3
frames agree, the winner has a clear majority, and one frame was genuinely
confident (score 0.60+). Everything weaker becomes "rest". The thresholds
are not hand-picked: a deterministic sweep tunes them against objects the
model should reject (categories that were dropped during relabeling, like
textiles and batteries).

Evaluation runs in **three tiers**: a normal validation split (the
optimistic number), the never-trained RealWaste set (the honest
generalization number), and a degraded copy of it that mimics cheap-camera
imaging (the robustness number).

## Results

The first full training run is **in progress right now** -- the pipeline is
built and tested end to end, and this table fills in as the run finishes.
Every number will be tied to a run config and seed in
`experiments/runs.jsonl`.

| Tier | What it measures | mAP50 | mAP50-95 |
|---|---|---|---|
| VAL | in-distribution validation split | _pending_ | _pending_ |
| TEST-1 (RealWaste) | generalization to a never-seen dataset | _pending_ | _pending_ |
| TEST-2 (degraded) | robustness to cheap-camera imaging | _pending_ | _pending_ |

| Rest-rule sweep | Value |
|---|---|
| Chosen thresholds (`tau_frame` / `min_votes` / `high_water`) | _pending_ |
| Wrong-class rate at chosen point | _pending_ |
| Rejection ("rest") rate at chosen point | _pending_ |

## Good to know

- **Reproducible by design.** Fixed seed (42), pinned dependencies,
  checksummed dataset downloads, deterministic training settings.
- **License:** AGPL-3.0-or-later (see `LICENSE`), inherited from the
  Ultralytics ecosystem. Dataset licenses are listed above.

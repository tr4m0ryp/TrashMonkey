# yolo-waste-sorter: Six-Class Waste Detection with Open-Set Rejection

## Project Overview

Most public waste-vision work stops at classification benchmarks. This
project packages the full path from raw public data to a deployable
detector: a fine-tuned **YOLO11n** (Ultralytics) model that localizes and
classifies single waste items into six material classes -- **plastic,
paper, cardboard, metal, glass, organic** -- and refuses to guess when it
is not confident. Anything the model cannot classify reliably is rejected
as **"rest"** rather than forced into a material class.

The model is trained entirely on **remapped public datasets** -- no
private photography. Five sources (TrashNet, Drinking Waste, Garbage
Detection, Recyclable & Household Waste, RealWaste) are downloaded,
label-remapped onto the six target classes, auto-boxed where only
classification labels exist, deduplicated, balanced, and merged into one
YOLO dataset. The repository deliberately **stops at the artifacts**: an
exported model in any supported Ultralytics format (ONNX by default;
TensorRT FP16 for NVIDIA edge devices, built on-device) plus the tuned
`thresholds.yaml`. Wiring those artifacts to cameras or control hardware
is an integration concern outside this repo's scope.

Two design choices carry the project. First, **"rest" is not a trained
class**: it is a multi-frame consensus rule applied downstream of the
detector, because a single confidence check provably leaks unknown
objects, while a voting rule requires the model to be fooled the same way
repeatedly. Second, **honest evaluation**: an entire source dataset
(RealWaste) is held out as an unseen test set, and a camera-degraded copy
of it estimates accuracy under low-cost imaging. Every reported number
ties to a run config and a fixed seed (42).

## How It Works

```
 5 public datasets          data pipeline (make repro)              training
+-----------------+   +--------------------------------------+   +-----------------+
| trashnet        |   | download -> remap -> autobox -> qa   |   | yolo11n.pt      |
| drinking-waste  |-->|  -> dedup -> balance -> split        |-->| AdamW, seeded,  |
| garbage-det.    |   | (Grounding DINO boxes, pHash dedup,  |   | camera-degrade  |
| household-waste |   |  QA gate halts on bad boxes)         |   | augmentation    |
| realwaste*      |   +--------------------------------------+   +--------+--------+
+-----------------+        *held out, never trained                       |
                                                                          |
                v-------------------------------------------------------+
+----------------------+   +----------------------+   +----------------------------+
| three-tier eval      |   | threshold tuner      |   | export                     |
| VAL / TEST-1 /       |-->| sweeps consensus     |-->| onnx / engine / ... +      |
| TEST-2 (degraded)    |   | rule -> thresholds.  |   | thresholds.yaml = the      |
+----------------------+   | yaml + sweep.csv     |   | deliverable artifacts      |
                           +----------------------+   +----------------------------+
```

1. **Data pipeline** (`make repro`) -- fetch the five sources into
   `data/raw/` (SHA-256 pinned), remap every source label onto the six
   classes via `configs/datasets.yaml`, auto-box classification-only images
   with **Grounding DINO** (BiRefNet and center-box fallbacks), run a QA gate
   that halts the pipeline for human review on suspect boxes, pHash-dedup
   across sources in fixed priority order, cap-balance classes, and emit one
   YOLO dataset. Dropped labels (e.g. textiles, "Miscellaneous Trash") become
   the **open-set probe pool** for threshold tuning -- never training data.
2. **Training** -- full fine-tune of COCO-pretrained `yolo11n.pt` with a
   pinned AdamW recipe (Ultralytics' `optimizer='auto'` silently flips to SGD
   with dataset size, so it is pinned). An Albumentations stack simulates
   low-cost camera sensors at train time -- JPEG blocking, ISO noise, motion
   blur, white-balance drift, downscaling -- so cheap-camera artifacts are
   learned, not feared. GPU runs go through `notebooks/manager.ipynb` on
   Colab; every run appends config, git hash, metrics, and runtime to
   `experiments/runs.jsonl`.
3. **Evaluation** -- three tiers: **VAL** (stratified 15% split, the
   optimistic number), **TEST-1** (leave-one-source-out RealWaste, the
   generalization number), and **TEST-2** (TEST-1 degraded at severities 1-5,
   the expected accuracy under degraded imaging).
4. **Threshold tuning** -- a deterministic sweep over the consensus
   parameters (per-frame tau, vote count, high-water mark) trades
   misclassification rate against rejection rate on degraded frames plus the
   open-set probe pool, then emits `thresholds.yaml` -- the deployment
   artifact that ships next to the exported model.
5. **Export** -- trained weights export to any supported Ultralytics format
   (ONNX default; TensorRT on-device) with a dummy-inference smoke test.
   The exported model plus `thresholds.yaml` are the repo's deliverables;
   consuming them in a live system is up to the integration.

### The consensus rule (open-set rejection without a rest class)

The rule is defined over a sequence of frames observing a single object. A
frame casts a **qualified vote** for class `c` when its top detection is `c`
with score >= `tau_frame`. The object is assigned class `c` only if all
three hold over its sightings:

| Condition | Default | Meaning |
|---|---|---|
| qualified votes for `c` >= `min_votes` | 3 | repeated agreement, not one lucky frame |
| `c` holds a strict majority of all qualified votes | -- | no split decisions |
| max single-frame score for `c` >= `high_water` | 0.60 | at least one confident look |

Anything else -> **rest**. Starting values (`tau_frame` 0.40, `conf_floor`
0.25) are sweep-tuned before deployment; the pure decision function lives in
`src/yolo_waste_sorter/models/thresholding/consensus.py`, and
`yolo_waste_sorter.deploy.load_threshold_params` is the matching fail-fast
reader an integration can use to consume `thresholds.yaml`.

## Data Sources

| Source | License | Images | Labels | Role |
|---|---|---|---|---|
| [TrashNet](https://github.com/garythung/trashnet) | MIT | 2,527 | cls, white posterboard | train |
| [Drinking Waste](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification) | CC0-1.0 | 4,828 | det (ships YOLO boxes) | train |
| [Garbage Detection](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection) | CC BY 4.0 | 10,464 | det, in-the-wild | train |
| [Recyclable & Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) | MIT | 15,000 | cls, 30 categories | train |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (UCI 908) | CC BY 4.0 | 4,752 | cls, landfill items | **TEST-1 holdout** |

The full per-label mapping (including resin-code collapses like PET/HDPE ->
plastic and the DROP routing) is `configs/datasets.yaml`; source order there
is the cross-dataset dedup priority.

## Quick Start

```bash
git clone <repo-url> && cd yolo-waste-sorter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]"

make test                  # unit + integration suite (pytest)
make lint                  # ruff + mypy --strict
FAKE_MODEL=1 make smoke    # offline end-to-end: pipeline -> train -> eval -> thresholds
```

The smoke harness (`python -m yolo_waste_sorter.smoke`) runs the entire
chain on synthetic fixtures in a throwaway tmp dir -- one `PASS <step>` line
per stage. With `ultralytics` installed it uses the real model; `FAKE_MODEL=1`
injects offline fakes (no downloads, no weights).

<details>
<summary>Real data pipeline (workstation)</summary>

```bash
pip install -e ".[boxing]"   # Grounding DINO + BiRefNet auto-boxing chain
make repro                   # download -> remap -> autobox -> qa -> dedup -> balance -> split
```

Kaggle sources need `~/.kaggle/kaggle.json` credentials. The QA stage exits
with code 2 when boxes need human review; inspect the report under
`data/interim/`, then rerun with `--ack-review`:

```bash
python -m yolo_waste_sorter.data.pipeline run --ack-review
```

`data/raw/` is never mutated after download; reruns are idempotent and
SHA-256-verified against `configs/datasets.yaml`.
</details>

<details>
<summary>GPU training (Colab)</summary>

Open `notebooks/manager.ipynb` in Colab and "Runtime > Run all". The notebook
is a manager only -- zero logic, it imports and calls the package (clone/
update, cache check, smoke test, checkpoint resume, train, evaluate, plot).
Locally:

```bash
python -m yolo_waste_sorter.models.train --smoke        # tiny CPU sanity run
python -m yolo_waste_sorter.models.train                # full run per configs/config.yaml
python -m yolo_waste_sorter.models.evaluate             # VAL / TEST-1 / TEST-2 report
python -m yolo_waste_sorter.models.thresholds           # sweep -> thresholds.yaml + sweep.csv
```
</details>

<details>
<summary>Export</summary>

Export trained weights to any supported Ultralytics format -- ONNX is the
portable default; TensorRT (`--format engine`) targets NVIDIA edge devices
and must run **on the deployment device** (engines are bound to the TensorRT
version and compute capability, so the export script refuses non-aarch64
hosts unless forced):

```bash
python -m yolo_waste_sorter.deploy.export --weights models/best.pt                  # ONNX
python -m yolo_waste_sorter.deploy.export --weights models/best.pt --format engine  # TensorRT FP16
```

Every export is smoke-tested with a dummy inference before the artifact is
trusted. The exported model and `thresholds.yaml` together are everything a
downstream system needs; camera capture and decision plumbing are not part
of this repository.
</details>

## Technical Details

| Path | Responsibility |
|---|---|
| `configs/config.yaml` | seed, classes, training recipe, augmentation, eval tiers, thresholds, deploy |
| `configs/datasets.yaml` | source registry: fetchers, SHA-256 pins, licenses, label mappings |
| `src/yolo_waste_sorter/data/download/` | fetchers (kaggle/http/local), manifest + checksum verification |
| `src/yolo_waste_sorter/data/remap/` | source labels -> six target classes, DROP routing to the probe pool |
| `src/yolo_waste_sorter/data/autobox/` | Grounding DINO -> BiRefNet -> center-box chain for cls-only sources |
| `src/yolo_waste_sorter/data/qa/` | box checks, cross-checks, human-review gate |
| `src/yolo_waste_sorter/data/pipeline/` | staged runner: download -> remap -> autobox -> qa -> dedup -> balance -> split |
| `src/yolo_waste_sorter/models/training/` | seeded fine-tune, smoke mode, run logging to `experiments/runs.jsonl` |
| `src/yolo_waste_sorter/models/evaluation/` | three-tier eval, severity curves, report artifact |
| `src/yolo_waste_sorter/models/thresholding/` | consensus rule, simulation, sweep, `thresholds.yaml` writer |
| `src/yolo_waste_sorter/deploy/` | multi-format model export + `thresholds.yaml` reader (artifacts only, no device I/O) |
| `src/yolo_waste_sorter/smoke/` | offline end-to-end harness (`FAKE_MODEL=1` supported) |
| `paper.tex` + `refs.bib` | technical report (NeurIPS template); `make paper` builds via tectonic |

Training recipe highlights (full values in `configs/config.yaml`): AdamW
`lr0=0.001 -> lrf=0.01`, 100 epochs, batch 16, `imgsz=640`,
`deterministic=true`, `cache=disk` (RAM caching breaks determinism),
`close_mosaic=10`, 180-degree rotation + flips (waste items have no
canonical orientation), `hsv_h` kept low at 0.015 because hue is the material
cue. Fallback if nano underperforms: `yolo11s.pt` -- but only after data
fixes, per the escalation policy in `models/training/escalation.py`.

Reproducibility: seed 42 everywhere (`utils/seed.py`), pinned dependency
ranges in `pyproject.toml`, `mypy --strict`, identical inputs produce
byte-identical `thresholds.yaml` and `sweep.csv`.

## Roadmap

- First full GPU training run and the three-tier numbers it produces (the
  pipeline, eval, tuner, and export are built and smoke-tested end to end).
- Threshold sweep on real degraded frames + open-set probes; freeze
  `thresholds.yaml`.
- Latency benchmarks of exported artifacts on reference edge hardware.

## Limitations and License

Single-object, plain-background presentation is a hard assumption -- this is
not a cluttered-scene detector. Training data is public-dataset only, so the
domain gap to any specific camera setup is managed (degradation-aware
augmentation, degraded test tier, consensus rejection) rather than
eliminated; the rest outcome exists precisely because the model will be
wrong sometimes. Latency figures are design estimates until measured on
target hardware.

Licensed under **AGPL-3.0-or-later** (see `LICENSE`) -- inherited from the
Ultralytics ecosystem. Dataset licenses and attributions are listed above and
in `configs/datasets.yaml`.

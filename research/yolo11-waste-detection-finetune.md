# YOLO11 Waste Detection Fine-Tune -- Technical Design
# Started: 2026-06-10
# Source vision: notes/yolo11-waste-detection-finetune.md

## Brief

Fine-tune yolo11n (COCO-pretrained, Ultralytics) for six-class waste detection
(plastic, paper, cardboard, metal, glass, organic) on white-background,
single-object, fixed-camera frames; train exclusively on remapped public
datasets; export TensorRT FP16 for a Jetson Orin Nano (Super) fed by three
ESP32-CAM streams. "rest" is a confidence-threshold rule in control logic, not
a class. This doc resolves the ten parked questions from the vision brief:
label mapping, dataset shortlist, bbox generation, balancing, augmentation,
val/test composition, fine-tune recipe, export chain, rest-threshold policy,
and dedup/licensing.

## Recommended Technical Design

Build one merged, deduplicated, license-attributed YOLO dataset (~5-10k images)
from clean-background public waste sets remapped to the six classes
[exact source mix and mapping table: T1/T2, pending census]. Classification-only
sources get boxes from **Grounding DINO** (class label from the source folder,
the prompt only localizes), with a BiRefNet-mask fallback and a flagged
center-box last resort, gated by automated QA plus a 10% stratified human
review (T3).

Fine-tune **yolo11n** from COCO weights with a pinned recipe -- AdamW lr0=0.001,
100 epochs, batch 16, imgsz 640, full fine-tune, cache=disk, seed 42 (T7) --
augmented for the known deployment camera: degrees=180/flipud for the top-down
view, plus an Albumentations stack simulating measured OV2640 defects (JPEG
blocking, ISO noise, motion blur, Planckian white-balance drift, downscale)
(T5). Training runs on Colab GPU through a **manager notebook** (thin
orchestrator over `src/yolo_waste_sorter/`, per project convention); all logic
stays in the package.

Evaluate on three tiers -- stratified instance-grouped VAL for selection,
leave-one-source-out TEST-1, and ESP32-degraded copies as TEST-2, the demo-day
proxy (T6). Escalate to yolo11s only if val mAP50 < 0.95 or any per-class
mAP50/recall < 0.90 after one data-fix iteration (T7).

Deploy on **JetPack 6.2.2**: export the TensorRT FP16 engine ON the Jetson at
imgsz 640, MAXN SUPER power mode (T8). Runtime is a single Python process:
three grab-latest daemon threads over the ESP32-CAM MJPEG streams (SVGA),
round-robin inference at ~5 ms/frame, and the rest-bin rule as **multi-frame
consensus** -- frame votes at tau>=0.40, sort on >=3 agreeing votes holding a
strict majority with a 0.60 high-water mark, else rest; thresholds tuned by
sweep on degraded validation including a wilderness (unknown-object) probe set
built from the census drop list (T9). The engine artifact ships with a
`thresholds.yaml` for control logic.

## Decisions

### T3: Bounding-box generation for classification-only datasets
**Decision:** **Grounding DINO (Swin-T) via `autodistill-grounding-dino`** as
primary auto-labeler: one prompt per material class, keep the single
highest-confidence box per image, threshold ~0.25-0.35. Class labels come from
the source dataset's folder, NOT the detector -- the prompt only localizes,
which removes categorization noise (the most damaging type) by construction.
Fallback chain per image: (1) Grounding DINO box; (2) BiRefNet mask ->
enclosing rect when DINO returns nothing; (3) fixed center box (image minus 5%
margin), flagged for manual review. Fallbacks will fire mostly on the
paper/white-plastic subset.
QA: automated checks on 100% of boxes (exactly-one-box, area-ratio <5% or >95%
flags, per-class area/aspect z-score >3, box touching >=3 edges, confidence
<0.3 -> review queue) + human review of a 10% stratified sample (oversampling
paper/white-plastic) and all flagged images, accept at IoU >= 0.8.
Acceptance: <=10% of sampled boxes failing review overall, <=20% on
localization-only grounds. Cross-check: measure IoU of our auto-boxes against
the human-annotated TrashNet on Roboflow Universe (Polygence, 2,524 images) --
target median IoU >= 0.8, reported in the paper.
**Why:** ~24% of TrashNet is paper on white posterboard -- the documented
failure case for every contrast-based method (Otsu collapse on non-bimodal
histograms, GrabCut mislabeling, rembg/U2-Net shadow-and-gap failures on
white-on-white -- F15). Grounding DINO detects semantically, not by contrast;
it is Apache-2.0 end-to-end, and auto-labels reach ~90-95% of human-label
downstream mAP on harder benchmarks than ours (F15). At 20% pure localization
noise the cost is only ~2 mAP, so the <=10% mixed-error budget keeps expected
degradation in the 1-2 mAP range (F16). ~0.3-0.4 s/image: the full corpus
labels in minutes on a T4.
**Alternatives rejected:** Otsu/contours/GrabCut as primary (white-on-white
collapse); rembg/U2-Net as primary (same); YOLO-World (GPL-3.0 -- viral
license); Grounding DINO 1.5 (closed weights, paid API -- kills
reproducibility); Roboflow Auto Label as a dependency (credit-metered,
forced-public data on the free tier); full-image boxes as primary (systematic
bias, not random noise -- model regresses frame-sized boxes; kept only as
flagged last-resort fallback). Note: Stanford precedents hand-cropped TrashNet
rather than auto-thresholding it -- no published classical pipeline succeeded
on raw TrashNet (F15).
**Confidence:** high.

### T5: Augmentation recipe (closing the public-data -> ESP32-CAM gap)
**Decision:** Two layers.
Native Ultralytics args: `degrees=180`, `flipud=0.5`, `fliplr=0.5` (top-down =
no canonical orientation), `hsv_v=0.5` (up from 0.4), keep `hsv_h=0.015`
(do NOT raise -- hue is the material cue), `hsv_s=0.7`, `scale=0.5`,
`translate=0.1`, `mosaic=1.0` with `close_mosaic=10`, `mixup=0`, `cutmix=0`,
`shear=0`, `perspective=0`, `copy_paste=0` (unusable: needs seg masks).
Albumentations layer (auto-engages when installed; pass `augmentations=[...]`
via the Python API -- it replaces only the Albumentations block, native augs
stay): `ImageCompression(quality_range=(50,85), p=0.5)`, `ISONoise(p=0.3)`,
`GaussNoise(p=0.2)`, `MotionBlur(blur_limit=(3,7), p=0.3)`,
`Defocus(radius=(1,3), p=0.1)`, `PlanckianJitter(p=0.3)` (physical white-balance
drift without destroying hue), `Downscale(scale_range=(0.4,0.75), p=0.3)`,
`RandomBrightnessContrast(p=0.2)`. Target: ~40-60% of samples get at least one
degradation; severities matched to measured OV2640 output (SVGA, mid JPEG
quality, 40 dB SNR).
**Why:** The deployment camera's defects are KNOWN (OV2640: JPEG blocking,
AGC noise, AWB green/warm drift, motion + fixed-focus blur, low effective
resolution -- F7), so we target them directly; corruption-style augmentation
demonstrably restores 30-60% performance lost under camera-quality shift (F8).
Mosaic stays high because public-set backgrounds are heterogeneous while deploy
is always white -- mosaic breaks background correlation; close_mosaic restores
single-object statistics for the final epochs.
**Alternatives rejected:** Raising hsv_h for color robustness (destroys
glass/plastic/organic hue discrimination -- use PlanckianJitter instead); mixup
(blends two materials = label noise; empirically hurts nano models, YOLOX
ablation); built-in copy_paste (segmentation-only); generic ImageNet-C severity
presets (severity-sensitive transfer -- match to measured camera output).
**Confidence:** high on the structure, medium on exact severity values --
calibrate once a real ESP32-CAM frame sample exists.

### T6: Validation/test composition with zero deployment images
**Decision:** Three tiers, all three reported in the paper.
1. **VAL** -- held-out split of the merged set, stratified by source-dataset x
   class, grouped by physical object instance (all photos of one item stay on
   one side). Drives model selection, early stopping, threshold sweeps. An
   optimistic ceiling, never a deployment claim.
2. **TEST-1 (dataset shift)** -- leave-one-SOURCE-out: an entire public dataset
   never trained on, chosen for deployment-like presentation (single object,
   clean background).
3. **TEST-2 (demo-day estimate)** -- degraded copies of TEST-1 through the
   ESP32 pipeline (downscale to 800x600 and back, JPEG re-encode q60-80,
   ISO/Gaussian noise, Planckian WB shift, mild motion blur) at 3-5 severity
   levels, reported as a curve.
   Never tune on TEST-1/TEST-2. The rest-bin confidence threshold is chosen on
   DEGRADED images, not clean ones -- corruption suppresses confidence and a
   clean-tuned threshold over-rejects at deploy.
**Why:** Random splits are "highly overoptimistic" vs unseen sources (F9);
training-domain validation is the most reliable practical selection criterion
absent target data (DomainBed); degraded-copy evaluation is established
practice (ImageNet-C / COCO-C, MMDetection robustness tooling -- F10) and
composes the two independent gap axes (unseen-source statistics x sensor
degradation). Instance-grouping prevents near-duplicate leakage from
TrashNet-style repeated shots of the same item.
**Alternatives rejected:** Plain random split (inflated numbers); compositing
objects onto white as the test set (weaker precedent than degraded-copy, and
compositing is already used on the training side); waiting for own-board
images (out of scope by C4).
**Confidence:** high. Calibrate the proxy with a small real ESP32 eval sample
the day hardware exists (calibration only, not tuning).

### T7: Fine-tuning recipe for yolo11n + escalation rule
**Decision:** Full fine-tune (no freezing) from `yolo11n.pt`, 100 epochs, explicit
`optimizer="AdamW", lr0=0.001, momentum=0.9`, `batch=16`, `imgsz=640`,
`patience=50`, `close_mosaic=10`, `cache="disk"`, `seed=42`, `deterministic=True`,
`amp=True`, `flipud=0.5` added to defaults. Select `best.pt` (fitness =
0.1·mAP50 + 0.9·mAP50-95), report test-split numbers.
Escalation rule: ship yolo11n iff val mAP50 >= 0.95 AND every per-class mAP50
>= 0.90 AND every per-class recall >= 0.90 (at conf 0.25). On failure, do ONE
data-fix iteration (label audit, per-class error review) first; only then
escalate to yolo11s with the identical recipe. If yolo11s also fails, the
dataset is the problem, not capacity.
**Why:** `optimizer='auto'` silently flips AdamW<->SGD at 10,000 iterations,
which straddles exactly our 5-10k-image x 100-epoch range -- pinning removes a
silent config change (F1). lr0=0.001 matches both the auto-computed value for
nc=6 and the Ultralytics fine-tuning guide. Full fine-tune: freezing costs up
to ~10 mAP50 (monotonic in trainable depth, F2) and saves nothing meaningful on
a 2.6M-param model. patience=50 not 10-20 so early stop cannot fire before the
close_mosaic fitness bump. cache="disk" because cache="ram" breaks determinism
even with fixed seed (F4). The 0.95/0.90 escalation bars are calibrated from
published clean-coarse waste fine-tunes reaching mAP50 0.92-0.99 (F5); the
per-class recall clause protects the rest-bin rule (a low-recall class leaks
into rest).
**Alternatives rejected:** `optimizer='auto'` (silent dataset-size dependency);
`freeze=11` backbone freeze (accuracy loss, negligible savings -- kept as one
ablation arm); `batch=-1` auto (hardware-dependent, not reproducible);
patience=10-20 (can stop before mosaic closes); smaller imgsz for speed (no
speed pressure: 4.57 ms/frame FP16 on the target, F6).
**Confidence:** high. Verified against ultralytics v8.3.0 source, not just docs.
Caveat: YOLO11's backbone is layers 0-10, so a backbone freeze is `freeze=11`,
not the `freeze=10` quoted in YOLOv8-era material (F2).

### T8: Export/deployment chain
**Decision:** Pin **JetPack 6.2.2** (L4T 36.5.0; CUDA 12.6, TensorRT 10.3.0,
cuDNN 9.3). Enable MAXN SUPER (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
Runtime via the official `ultralytics/ultralytics:latest-jetson-jetpack6`
docker image (native install with Jetson torch wheels as fallback). Export
**on the Jetson**: `yolo export model=best.pt format=engine half=True
imgsz=640` -- TensorRT engines are non-portable across TRT versions and compute
capabilities (Orin is SM 8.7). FP16, static batch=1.
Ingest: cameras at SVGA 800x600 (~8-12 fps stock firmware, ~27 KB JPEG), one
grab-latest daemon thread per camera (continuous `cap.grab()`, keep newest --
`CAP_PROP_BUFFERSIZE` is unreliable on HTTP streams), single inference loop
round-robins the three latest frames on one engine. Plain Python; no
DeepStream, no Triton, no GStreamer. Dedicated 2.4 GHz AP for the cams, solid
5 V supply (undervoltage measurably drops fps).
**Why:** yolo11n TensorRT FP16 is ~4.9 ms/frame on this exact board (F6/F11) --
the ESP32-CAM at ~8 fps is the bottleneck, not the Orin; the GPU could serve
each camera at 13-22 Hz. End-to-end glass-to-decision is ~90-160 ms typical
(~0.5 s worst with WiFi jitter), so a 5-frame consensus completes in ~0.6 s
against the 1-2 s conveyor budget. JetPack 7.x ecosystem (torch wheels,
containers) had not caught up for Orin Nano as of mid-2026. DeepStream pays
off at tens of hardware-decodable RTSP streams; 3x MJPEG/HTTP at 8 fps with a
5 ms model gives it nothing to optimize, and it does not natively ingest
HTTP-MJPEG anyway.
**Alternatives rejected:** INT8 (saves ~1 ms against a >=1000 ms budget; needs
>=500-image on-device calibration; TRT 10.3 has an INT8+end2end build bug --
revisit only for power/thermals); UXGA capture (1.29 fps -- blows the budget);
DeepStream/Triton (overhead without benefit at this scale); exporting the
engine on a desktop GPU (engines don't deserialize across SM/TRT).
**Confidence:** high. Headroom note: yolo11s at 7.3 ms FP16 also fits if T7
escalation triggers.

### T9: Confidence-threshold policy for the rest-bin rule
**Decision:** Multi-frame consensus, not a single high threshold.
Per frame: run at permissive `conf=0.25`. A frame casts a qualified vote for
class c if its top detection is c with score >= tau_frame (start 0.40).
Sort to bin c iff over the object's 5-15 sightings: (a) >= 3 qualified votes
for c, (b) c holds a strict majority of qualified votes, (c) max single-frame
score for c >= 0.60 (high-water gate). Otherwise -> rest bin.
Thresholds start global; switch to per-class iff the confidence achieving
precision >= 0.95 spans more than ~0.1 across classes on the val curves
(likely under imbalanced remapped data). No calibration in v1 (cost-curve
tuning absorbs miscalibration).
Tuning procedure: build a val sim including a "wilderness" set of non-class
objects; grid-sweep tau_frame x K x high-water threshold; plot wrong-bin rate
vs rest-rate; pick the Pareto knee subject to wrong-bin <= 1-2%. Tune on
DEGRADED (TEST-2-style) frames and re-verify on the exported FP16 engine --
quantization shifts score distributions.
**Why:** YOLOv8/11 has no objectness branch -- the score IS the per-class
sigmoid, miscalibrated and overconfident (F12). Single-frame thresholding is a
leaky open-set filter: naive score rejection lets 71-81% of out-of-distribution
objects through at 95% TPR (F13). Consensus converts per-frame error to
per-object error roughly binomially, so a LOWER per-frame threshold + agreement
rule dominates one high threshold: same wrong-bin protection, lower rest-rate.
Direct conveyor precedent: YOLOv8 + per-track majority vote (F14).
**Alternatives rejected:** Single high global threshold (worse rest-rate at
equal contamination); energy-score OOD heads (needs pre-sigmoid logits --
invasive on a TensorRT pipeline; deferred); training a 7th "other" class
(rejected at vision level, C3); temperature scaling in v1 (doesn't change
ranking; only needed if logic starts averaging confidences).
**Confidence:** high on the rule shape, medium on the starting values
(0.40/3/0.60) -- they are explicitly sweep-tuned in validation.

## Stack & Libraries

| Component | Choice | Call | License | Health note |
|---|---|---|---|---|
| Detection framework | `ultralytics >=8.3,<9` (pinned in pyproject) | Adopt | **AGPL-3.0** -- note in paper; fine for a course project | Mainline, very active |
| Train-time degradation | `albumentations >=1.0.3` | Adopt | MIT | Auto-engages in ultralytics when installed; custom list via Python API |
| Auto-boxing | `autodistill` + `autodistill-grounding-dino` | Adopt | Apache-2.0 (incl. weights) | One-off labeling step, not a runtime dep |
| Bbox fallback | `rembg[birefnet]` (BiRefNet weights) | Compose | MIT | Fires only where DINO returns nothing |
| Label QA (post-train pass) | `cleanlab` ObjectLab | Compose (optional) | AGPL-3.0 | Ranks worst boxes after first training round |
| Dedup | perceptual hashing via `imagehash` (pHash) | Adopt | BSD-2 | Exact method confirmed against census overlap findings (T10) |
| Dataset fetch | `kaggle` CLI + direct URLs; Roboflow Universe export for the human-annotated TrashNet cross-check only | Compose | -- | No Roboflow account dependency in the pipeline |
| Jetson runtime | JetPack 6.2.2; `ultralytics:latest-jetson-jetpack6` docker; torch 2.10/torchvision 0.25 Jetson wheels if native | Adopt | -- | Engine tied to TRT 10.3.0 + SM 8.7 |
| Stream ingest | `opencv-python` + threads (turbojpeg optional) | Build (small) | Apache-2.0 | Grab-latest reader is ~50 lines |
| Experiment tracking | `runs.jsonl` per run + ultralytics run dirs | Build (small) | -- | Per project convention; no external tracker |
| Dev | pytest, ruff, mypy (already pinned) | Adopt | -- | -- |

Training hardware: Colab GPU via manager notebook; smoke test on local CPU
first (project convention).

## Architecture

```
data pipeline (CLI, src/yolo_waste_sorter/data/)
  download --> remap (T1 table) --> autobox (T3: DINO->BiRefNet->centerbox)
    --> qa (auto checks + review queue) --> dedup (pHash, T10)
    --> balance (T4 caps) --> split (T6: VAL/TEST-1 instance-grouped, stratified)
    --> emit YOLO dataset + dataset.yaml + license/attribution table (paper)

training (src/yolo_waste_sorter/models/ + notebooks/manager.ipynb)
  configs/*.yaml --> train (T7 recipe, T5 augmentation) --> best.pt
    --> runs.jsonl + per-epoch val curves

evaluation (src/yolo_waste_sorter/models/)
  best.pt --> VAL / TEST-1 / TEST-2(degraded, shared transform code with T5)
    --> per-class P/R/mAP + confidence curves --> escalation check (T7 rule)

threshold tuner (src/yolo_waste_sorter/models/)
  best.pt + degraded VAL + wilderness probe --> consensus simulation
    --> sweep (tau_frame x K x high-water) --> Pareto knee --> thresholds.yaml

deployment (scripts on the Jetson)
  best.pt --> on-device export --> best.engine (FP16, 640, batch 1)
  3x ESP32-CAM MJPEG --> grab-latest threads --> round-robin infer
    --> per-object vote aggregation (T9 rule, thresholds.yaml)
    --> control-logic handoff: (class, confidence, timestamp) per decision
```

Key contracts: the degradation transform module is shared between training
augmentation (T5), TEST-2 generation (T6), and threshold tuning (T9) so all
three see the same camera model; `thresholds.yaml` + `best.engine` is the
complete deployment artifact; every reported number traces to a run config in
`runs.jsonl` (paper-scribe requirement). The detector-side handoff emits class,
confidence, and frame timestamp per decision -- the exact contract with the
control-logic team is an open thread inherited from the vision doc.

## Decisions Made For You (override in /refine)

1. **Pinned AdamW recipe over `optimizer='auto'`** -- auto flips optimizer with
   dataset size (F1). Change if you'd rather track Ultralytics defaults.
2. **Full fine-tune, no backbone freeze** -- alternative: `freeze=11` trains
   faster at some accuracy cost. Change if GPU budget gets tight.
3. **Grounding DINO as primary auto-boxer** -- alternative: BiRefNet-first
   (lighter, no prompt) or hand-labeling (~2,500 images). Change if you'd
   rather not depend on a 700MB vision-language model for a one-off step.
4. **degrees=180 + flipud=0.5** -- assumes truly top-down presentation. Change
   (degrees~15) if the cameras end up mounted at an angle.
5. **Three-tier eval with leave-one-source-out** -- costs one whole dataset's
   training images. Change to a plain stratified split if data turns out
   scarcer than expected after balancing.
6. **Multi-frame consensus (0.40/3-votes/0.60 high-water) over a single
   threshold** -- requires the control loop to track per-object votes. Change
   to a single conf=0.55 gate if control logic must stay single-frame.
7. **JetPack 6.2.2 + docker runtime** -- native install is the alternative if
   docker overhead annoys on the 8GB board.
8. **FP16, no INT8** -- INT8 saves ~1 ms but needs on-device calibration and
   hits a TRT 10.3 export bug. Change only if power/thermals demand it.
9. **Colab + manager notebook for training** -- per your ESPResso-V2
   convention. Change if a local/remote GPU box materializes.
10. **`cache='disk'`, `batch=16` fixed** -- reproducibility over per-machine
    speed auto-tuning.

## Key Findings

### F1: optimizer='auto' silently switches AdamW->SGD at 10k iterations
**Finding:** In ultralytics v8.3.0, `optimizer='auto'` ignores lr0/momentum and
picks SGD(0.01) if `ceil(n_images/max(batch,nbs=64)) * epochs > 10000`, else
AdamW(lr=round(0.002*5/(4+nc),6)=0.001 for nc=6). 5k imgs x 100 ep = 7.9k iters
(AdamW); 10k imgs x 100 ep = 15.7k iters (SGD) -- our planned dataset size range
straddles the flip point.
**Evidence:** v8.3.0 `ultralytics/engine/trainer.py` (read directly from the tag).
**Implications:** Pin `optimizer="AdamW", lr0=0.001, momentum=0.9` explicitly so
growing the dataset cannot silently change the recipe.

### F2: Full fine-tune beats freezing; YOLO11 backbone freeze index is 11
**Finding:** Unfreezing depth monotonically improves accuracy (~+10 mAP50
head-only -> near-full on a comparable 6-class YOLOv8n fine-tune); Ultralytics
guidance maps "small-medium dataset, similar domain" to full fine-tune from
pretrained. YOLO11's backbone spans modules 0-10 (SPPF=9, C2PSA=10), so a
backbone freeze is `freeze=11` -- the commonly quoted `freeze=10` leaves C2PSA
trainable.
**Evidence:** arXiv:2505.01016; docs.ultralytics.com/guides/finetuning-guide;
v8.3.0 `cfg/models/11/yolo11.yaml`.
**Implications:** `freeze=None` for the main run; `freeze=11` only as ablation.

### F3: best.pt and early stopping track fitness = 0.1*mAP50 + 0.9*mAP50-95
**Finding:** Ultralytics checkpoints best.pt and triggers patience on a weighted
fitness dominated by mAP50-95, not raw mAP50.
**Evidence:** v8.3.0 `ultralytics/utils/metrics.py::Metric.fitness`,
`engine/trainer.py`.
**Implications:** patience must outlast the close_mosaic window (fitness bumps
when mosaic turns off at epoch 90); report from best.pt on the held-out test
split, never the selection split.

### F4: cache='ram' breaks training determinism even with fixed seed
**Finding:** RAM caching's ThreadPool fill order and mosaic-buffer order are
nondeterministic across workers; disk cache is not affected. Bitwise repro only
holds within identical GPU/driver/torch; expect ~±0.2-0.5 mAP50-95 run noise
across hardware.
**Evidence:** ultralytics issues #15960, #21783; pytorch randomness notes.
**Implications:** `cache="disk"` for tracked runs; record GPU+versions per run;
treat sub-noise-floor recipe deltas as ties.

### F5: Clean coarse waste fine-tunes publish mAP50 0.90-0.99; long-tail cluttered sets 0.12-0.55
**Finding:** Nano/small YOLO on clean, coarse-class waste data reaches mAP50
0.92-0.99 (e.g. YOLOv8s 92.7 vs YOLOv8l 93.4 on 3 coarse classes -- the n->l gap
collapses below 1 point on easy tasks). TACO-style long-tail cluttered data
yields 0.12-0.37 (YOLOv8n/s) and WaRP 28-class conveyor 0.55.
**Evidence:** IEEE 10696214; PMC11244501; github junaidzeb123/trash-detection-
yolo-unet; IIETA RIA 38(4).
**Implications:** Calibrates the T7 escalation bars (0.95/0.90) and confirms
nano-first is sound; sub-0.90 val mAP50 signals a data problem, not capacity.

### F6: yolo11n TensorRT FP16 on Orin Nano Super = 4.91 ms/frame at 640
**Finding:** Official Ultralytics Jetson benchmarks (Orin Nano Super, JetPack
6.1, imgsz 640): yolo11n engine FP32 7.60 ms, FP16 4.91 ms, INT8 3.91 ms;
yolo11s FP16 7.30 ms. (The 4.57 ms figure floating in current docs is YOLO26n,
not yolo11n.) Published INT8 mAP collapse in the YOLO11 table is a COCO8
calibration artifact, not a real INT8 property.
**Evidence:** docs.ultralytics.com/guides/nvidia-jetson/ (v8.3.100 docs
snapshot for the YOLO11 table).
**Implications:** No speed pressure at 640 -- keep train=infer=640; INT8 is the
later lever, never imgsz reduction; yolo11s (7.3 ms) also fits the budget if
escalation triggers.

### F7: The OV2640 deployment signature is known and simulable
**Finding:** ESP32-CAM (OV2640): SVGA 800x600 streams at ~8.25 fps measured
(stock firmware, ~27 KB JPEG); UXGA only 1.29 fps. SNR 40 dB, AGC noise under
indoor light, on-chip JPEG with visible blocking at streaming quality, AWB
green/warm drift ("first frames dark and greenish"), fixed-focus ~65 degree
lens, rolling shutter. Undervoltage (<2.985 V) measurably degrades fps.
**Evidence:** OV2640 datasheet; arXiv:2505.24081 (measured fps benchmark);
Random Nerd Tutorials settings guide; espressif/esp32-camera issues #150/#185.
**Implications:** Train-time degradation stack (T5) and the degraded-copy eval
(T6) target these exact defects; cameras run SVGA, never UXGA; solid 5 V supply
and a dedicated 2.4 GHz AP are deployment requirements.

### F8: Corruption-style augmentation demonstrably closes camera-quality gaps
**Finding:** Detectors lose 30-60% of clean performance under noise/blur/
digital corruption; corruption-targeted training augmentation substantially
restores it, and tuned Gaussian/speckle noise augmentation is a SOTA-level
robustness baseline. Severity tuning matters: over-strong corruptions fail to
generalize -- but our deployment corruptions are KNOWN, so we target them.
**Evidence:** Michaelis et al. arXiv:1907.07484; Rusak et al. arXiv:2001.06057;
Dodge & Karam arXiv:1604.04004; Kitware nrtk-albumentations.
**Implications:** The T5 Albumentations stack is evidence-backed, with
severities matched to F7 measurements rather than generic presets.

### F9: Random-split validation is highly overoptimistic vs unseen sources
**Finding:** Train/test on the same source inflates scores (dataset-bias
literature since Torralba & Efros 2011); multi-source studies call random-split
estimates "highly overoptimistic" vs leave-source-out; waste-specific reviews
document TACO/TrashNet domain shift and call cross-dataset eval essential.
With no target-domain data, training-domain validation is the standard
model-selection criterion (DomainBed).
**Evidence:** arXiv:2403.15012; arXiv:2007.01434 (DomainBed); PMC12115937
(waste review); arXiv:2503.02241.
**Implications:** The three-tier eval in T6: VAL (selection ceiling), TEST-1
(leave-one-source-out), TEST-2 (degraded copies).

### F10: Degraded-copy evaluation is established deployment-proxy practice
**Finding:** ImageNet-C / Pascal-C / COCO-C are exactly "corrupted copies of
the val set as robustness proxy"; MMDetection ships tooling for it; a 2024
IJCV benchmark builds corruptions from real camera-sensor/ISP damage models.
**Evidence:** Hendrycks & Dietterich arXiv:1903.12261; Michaelis et al.
arXiv:1907.07484; Springer s11263-024-02096-6.
**Implications:** TEST-2 has direct precedent; report 3-5 severity levels as a
curve; state the proxy-imperfection caveat in the paper.

### F11: The ESP32-CAM is the system bottleneck, not the Orin
**Finding:** Per-frame on-Jetson cost is ~12-25 ms (decode + preprocess +
4.91 ms infer + post), so a 3-camera round-robin cycle is ~45-75 ms and the
GPU could serve each camera at 13-22 Hz -- but stock cameras produce ~8 fps.
End-to-end glass-to-decision: ~90-160 ms typical, ~0.5 s worst (WiFi jitter).
A 5-frame consensus completes in ~0.6 s against the 1-2 s conveyor budget.
**Evidence:** arXiv:2505.24081 (camera fps); Ultralytics Jetson benchmarks;
turbojpeg decode measurements.
**Implications:** No exotic runtime needed (plain Python + threads); 8-16
fresh frames per camera per conveyor window makes the T9 consensus rule
comfortable; camera firmware/network, not the model, is where latency risk
lives.

### F12: YOLOv8/11 confidence is a raw per-class sigmoid with no objectness
**Finding:** The v8/v11 anchor-free decoupled head has no objectness branch;
inference confidence is the per-class sigmoid score (multi-label, not
softmax-normalized). Detectors are systematically miscalibrated/overconfident;
per-class calibration differs under class imbalance (+1.7 AP on LVIS from
per-class calibration).
**Evidence:** ultralytics/nn/modules/head.py (source); maintainer confirmations
in issues #6214/#3707; arXiv:2210.02935; arXiv:2004.13546; arXiv:2102.01066.
**Implications:** A fixed threshold is not a probability statement; per-class
thresholds become worthwhile when per-class calibration diverges (T9 decision
criterion: >0.1 spread in the precision>=0.95 confidence across classes).

### F13: Naive score thresholding leaks 71-81% of unknown objects
**Finding:** At 95% true-positive rate on known classes, max-score rejection
lets 70.99-80.94% of out-of-distribution objects through (FPR95, VOS
benchmark); unknowns are often confidently misclassified as the nearest known
class. Energy scores improve this but need pre-sigmoid logits (invasive on
TensorRT).
**Evidence:** Dhamija et al. WACV 2020 (IEEE 9093355); VOS ICLR 2022
arXiv:2202.01197; arXiv:2412.20701.
**Implications:** Single-frame thresholding cannot be the rest-bin mechanism;
the multi-frame consensus rule (T9) is the strongest stock-pipeline defense;
the rest bin must be sized assuming imperfect rejection -- a wilderness probe
set goes into validation.

### F14: Multi-frame majority voting has direct conveyor precedent and quantified gains
**Finding:** YOLOv8 + ByteTrack + per-track majority vote is published for
conveyor inspection (frame predictions fluctuate under motion blur/lighting;
track-level aggregation stabilizes); K-frame voting lifted accuracy 89->91%
and F1 0.80->0.83 (K=10->50) with diminishing returns; consensus converts
per-frame error to per-object error roughly binomially.
**Evidence:** arXiv:2602.19278; arXiv:2503.04139; arXiv:1706.03309;
docs.ultralytics.com/modes/track/.
**Implications:** Lower per-frame threshold + agreement rule dominates one
high threshold (same contamination protection, lower rest-rate); on a
single-file belt, frame-to-object association is trivial without a tracker.

### F15: Contrast-based bbox methods fail white-on-white; Grounding DINO does not
**Finding:** ~24% of TrashNet is paper on white posterboard. Otsu collapses on
non-bimodal histograms; GrabCut mislabels low-contrast fg/bg; rembg/U2-Net
leaves shadows/gaps on white-on-white (documented). No published classical
pipeline succeeded on raw TrashNet -- Stanford precedents hand-cropped instead.
Grounding DINO (Apache-2.0, ~0.3-0.4 s/image) localizes semantically;
auto-labels reach ~90-95% of human-label downstream mAP on benchmarks harder
than single-centered-object data. Human-annotated TrashNet detection versions
exist on Roboflow Universe (Polygence 2,524 imgs) as free cross-check ground
truth.
**Evidence:** forum.opencv.org white-on-white thread; rembg discussion #566;
Cloudflare bg-removal eval (BiRefNet IoU 0.87); Voxel51 auto-labeling eval;
DART pipeline arXiv:2407.09174; CS229/CS230 TrashNet reports.
**Implications:** T3's primary/fallback chain and the free IoU cross-check.

### F16: Detection training tolerates ~20% localization noise (~2 mAP cost)
**Finding:** At 20% noise ratio, localization noise costs only -2.0 mAP --
mildest noise type after class noise (-2.4); but noise types compound
synergistically (-10.8 mAP at 20% combined); ~30% box noise roughly halves
mAP. Mild random box noise can even help (NBBOX). Systematic bias (fixed
full-image boxes) is different: the model regresses frame-sized boxes.
**Evidence:** arXiv:2312.13822 (Universal Noise Annotation); arXiv:2409.09424;
ResearchGate 354111821.
**Implications:** The T3 QA budget (<=10% mixed errors, <=20%
localization-only) keeps expected degradation at 1-2 mAP; class labels coming
from source folders (not the detector) removes the worst noise type entirely.

### F17: TrashNet composition quantified
**Finding:** TrashNet: 2,527 images, 512x384, single hand-placed object on
white posterboard -- glass 501, paper 594, cardboard 403, plastic 482, metal
410, trash 137. MIT license.
**Evidence:** github.com/garythung/trashnet.
**Implications:** Closest analogue to deployment presentation; the "trash"
class (137) maps to nothing in our six and is dropped; per-class counts feed
the T4 balancing math.

## References

### R1: Ultralytics v8.3.0 source (trainer, metrics, model cfg, seeds)
**Source:** https://github.com/ultralytics/ultralytics/tree/v8.3.0
**Takeaway:** Ground truth for T7 under our `<9` pin: auto-optimizer flip logic,
fitness weights, yolo11 backbone indices, seed handling. Current docs describe
YOLO26-era code -- always verify against the tag.

### R2: Ultralytics NVIDIA Jetson guide
**Source:** https://docs.ultralytics.com/guides/nvidia-jetson/
**Takeaway:** Official Orin Nano Super benchmarks (yolo11n FP16 4.91 ms),
docker images, export path, DLA absence on Orin Nano, best practices.

### R3: Ultralytics fine-tuning + tips guides
**Source:** https://docs.ultralytics.com/guides/finetuning-guide ;
https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/
**Takeaway:** Freeze-depth guidance, lr0=0.001 AdamW for fine-tunes, dataset
size bars (>=1500 imgs/class recommended), pretrained-weights guidance.

### R4: JetPack 6.2.x / Super mode (NVIDIA)
**Source:** https://developer.nvidia.com/embedded/jetpack-sdk-62 ;
https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/
**Takeaway:** Version pins (CUDA 12.6/TRT 10.3/cuDNN 9.3), `nvpmodel -m 2`
MAXN SUPER, throttling caveat.

### R5: TensorRT engine compatibility (NVIDIA)
**Source:** https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
**Takeaway:** Engines are tied to TRT version and device compute capability --
export must happen on the Jetson.

### R6: ESP32-CAM measured streaming benchmark
**Source:** https://arxiv.org/html/2505.24081v1
**Takeaway:** SVGA 8.25 fps / ~27 KB JPEG measured on stock firmware; UXGA
1.29 fps. The camera, not the Orin, is the system bottleneck.

### R7: Grounding DINO + autodistill
**Source:** https://arxiv.org/abs/2303.05499 ; https://docs.autodistill.com/
**Takeaway:** Open-vocabulary boxes from text prompts, Apache-2.0; the T3
primary labeler. Auto-labels reach ~90-95% of human-label downstream mAP
(https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance).

### R8: Universal Noise Annotation (label-noise tolerance)
**Source:** https://arxiv.org/html/2312.13822v1
**Takeaway:** 20% localization noise costs only ~2 mAP; mixed noise compounds;
grounds the T3 QA acceptance bars.

### R9: VOS / open-set detection benchmark
**Source:** https://ar5iv.labs.arxiv.org/html/2202.01197 ;
https://ieeexplore.ieee.org/document/9093355/
**Takeaway:** Score thresholding alone leaks 71-81% of OOD objects at 95% TPR;
unknowns get confident wrong detections -- why T9 uses consensus.

### R10: Conveyor multi-frame voting precedents
**Source:** https://arxiv.org/html/2602.19278v1 ; https://arxiv.org/pdf/2503.04139
**Takeaway:** Track-level majority voting stabilizes fluctuating per-frame
predictions on conveyors; quantified gains, diminishing returns past K~50.

### R11: ImageNet-C / robustness benchmarking lineage
**Source:** https://arxiv.org/abs/1903.12261 ; https://arxiv.org/abs/1907.07484
**Takeaway:** Degraded-copy evaluation is established practice; detectors lose
30-60% under corruption; corruption-targeted augmentation restores much of it.
Grounds TEST-2 and the T5 stack.

### R12: DomainBed / leave-source-out evidence
**Source:** https://arxiv.org/abs/2007.01434 ; https://arxiv.org/html/2403.15012v1
**Takeaway:** Random splits are overoptimistic; training-domain validation is
the standard selection criterion without target data. Grounds the T6 tiers.

### R13: OV2640 datasheet + ESP32-CAM settings guide
**Source:** https://www.uctronics.com/download/cam_module/OV2640DS.pdf ;
https://randomnerdtutorials.com/esp32-cam-ov2640-camera-settings/
**Takeaway:** Sensor SNR/AWB/JPEG characteristics that the T5 degradation
stack and TEST-2 simulate.

### R14: TrashNet
**Source:** https://github.com/garythung/trashnet
**Takeaway:** 2,527 single-object white-posterboard images, 6 classes, MIT --
the closest public analogue to our deployment presentation.

### R15: YOLOv8/11 head source (no objectness branch)
**Source:** https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py
**Takeaway:** Confidence = per-class sigmoid only; what the T9 threshold
actually thresholds.

<!-- Census references (dataset pages, licenses, overlap evidence) land here
     as R16+ when the census agent reports. -->

## Discarded Approaches

- **Own data collection / own-board val split** -- rejected at vision level (C4).
- **Trained 7th "other" class** -- rejected at vision level (C3); the rest rule
  is control logic. Research adds: the census drop list (unmappable categories)
  becomes the wilderness probe set for threshold tuning instead -- the rejects
  are repurposed, not trained on.
- **YOLO26** -- rejected at vision level (C1); also our `<9` pin predates its
  trainer changes (MuSGD), so recipes here are verified against 8.3.x.
- **Classical CV auto-boxing (Otsu/GrabCut/edges) as primary** -- documented
  white-on-white failures on ~24% of the closest dataset (F15).
- **rembg/U2-Net as primary boxer** -- same white-on-white failure mode (F15).
- **YOLO-World / Grounding DINO 1.5 / Roboflow Auto Label** -- GPL-3.0 /
  closed weights / paid lock-in respectively (T3).
- **Single high confidence threshold as the rest rule** -- leaks 71-81% of
  unknowns at matched recall; dominated by consensus (F13, F14).
- **Energy-score OOD detection** -- needs pre-sigmoid logits; invasive on a
  TensorRT pipeline; deferred, not needed at this difficulty.
- **DeepStream / Triton / GStreamer runtime** -- nothing to optimize at
  3 streams x 8 fps with a 5 ms model (T8).
- **INT8 export as default** -- ~1 ms saved against a ~1 s budget; calibration
  burden + TRT 10.3 bug (T8).
- **JetPack 7.x** -- ecosystem not caught up for Orin Nano mid-2026 (T8).
- **Mixup/cutmix augmentation; raised hue jitter** -- material-cue destruction,
  nano-scale harm evidence (T5).

## Risks & Open Threads

- [ ] **T1/T2/T4/T10 pending dataset census** (agent running) -- label-mapping
  table, source shortlist, per-class targets, dedup/license table. Blocks
  /readyforlaunch for the data-pipeline tasks only.
- [ ] **Organic-class thinness** -- the weakest class in public sets; census
  will quantify extractable counts; remedy options land in T4.
- [ ] **White-paper-on-white bbox QA burden** -- fallback chain fires most on
  the paper class; the 10% review oversamples it; if review fails the
  acceptance bar, escalate that subset to manual boxes (bounded: ~600 images).
- [ ] **Demo-day lighting drift vs white-background bet** -- vision-level risk;
  monitored at integration, mitigated by hsv_v/PlanckianJitter augmentation
  and the TEST-2 severity curve (worst-severity numbers = early warning).
- [ ] **Control-logic handoff contract** -- detector emits (class, confidence,
  timestamp) per decision; exact format/transport with the control team still
  open (inherited from vision doc; not a training question).
- [ ] **WiFi jitter worst case (~0.5 s)** -- eats consensus frames; mitigated
  by dedicated AP; if it persists, drop K from 3 to 2 at a slightly higher
  tau_frame (re-sweep).
- [x] Mosaic vs single-object deployment tension -- resolved: keep mosaic,
  close_mosaic=10 (T5).
- [x] Per-class vs global threshold -- resolved: criterion-based switch (T9).
- [x] yolo11n capacity worry -- resolved: published clean-coarse results +
  escalation rule (F5, T7).

## Build Plan

Phases ordered by dependency; P2-P4 parallelize after P1. Built for
/readyforlaunch task decomposition. All logic in `src/yolo_waste_sorter/`;
notebooks orchestrate only (project convention).

**P1 -- Data pipeline core** (blocked on census: T1 mapping, T2 shortlist)
- `data/download.py`: fetchers per source (kaggle CLI, direct), checksums,
  into `data/raw/` (never mutated).
- `data/remap.py`: T1 label-mapping table as data (YAML), drop-list handling;
  drops routed to `data/interim/wilderness/` for the T9 probe set.
- `data/autobox.py`: DINO -> BiRefNet -> centerbox chain (T3), writes YOLO
  labels + provenance.
- `data/qa.py`: automated checks, review queue emission, IoU cross-check
  against the human-annotated TrashNet.
- `data/dedup.py` + `data/balance.py` + `data/split.py`: pHash dedup (T10),
  caps (T4), instance-grouped stratified splits (T6), final `dataset.yaml` +
  attribution table for the paper.
- Wire `make repro` to run the chain end to end, seeded.

**P2 -- Training** (needs P1 output format only; recipe is fixed)
- `models/train.py`: T7 recipe from `configs/`, T5 Albumentations stack,
  runs.jsonl logging, smoke-test mode.
- `notebooks/manager.ipynb`: Colab orchestrator per CLAUDE.md convention
  (preflight, cache, smoke, resume, canary, train, eval, plots).
- `utils/degrade.py`: the shared ESP32 degradation transforms (T5/T6/T9).

**P3 -- Evaluation + threshold tuning** (needs P2 artifacts)
- `models/evaluate.py`: three-tier eval, per-class curves, escalation check.
- `models/tune_thresholds.py`: consensus simulation over VAL+wilderness,
  sweep, Pareto knee, emit `thresholds.yaml`.
- `visualization/`: per-class PR/confidence plots, TEST-2 severity curves ->
  `reports/figures/`.

**P4 -- Deployment** (needs best.pt; Jetson-side, independent of P3 tuning
until thresholds.yaml lands)
- Jetson setup script (JetPack 6.2.2 checks, power mode, docker pull).
- On-device export script + engine smoke test.
- `deploy/runtime.py`: stream readers, round-robin loop, vote aggregator,
  control-logic emit. Bench end-to-end latency against the F11 budget.

**P5 -- Paper integration** (continuous; scribe is active)
- Label-mapping + license tables (P1 outputs), recipe + results tables
  (P2/P3), deployment numbers (P4), limitations (proxy-eval caveat, F10).

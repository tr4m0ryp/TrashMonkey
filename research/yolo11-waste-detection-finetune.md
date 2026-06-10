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

<!-- pending -- filled at wrap-up -->

## Decisions

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

<!-- pending -->

## Architecture

<!-- pending -->

## Decisions Made For You (override in /refine)

<!-- pending -->

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

## References

<!-- R# entries -->

## Discarded Approaches

<!-- pending -->

## Risks & Open Threads

<!-- pending -->

## Build Plan

<!-- pending -->

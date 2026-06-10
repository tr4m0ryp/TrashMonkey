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

### F6: yolo11n TensorRT FP16 on Orin Nano Super = 4.57 ms/frame at 640
**Finding:** Official Ultralytics Jetson benchmarks: yolo11n engine FP32 7.53 ms,
FP16 4.57 ms, INT8 3.80 ms at imgsz 640 on Orin Nano Super.
**Evidence:** docs.ultralytics.com/guides/nvidia-jetson/.
**Implications:** No speed pressure at 640 -- keep train=infer=640; INT8 is the
later lever, never imgsz reduction; yolo11s (~8 ms) also fits the budget if
escalation triggers.

## References

<!-- R# entries -->

## Discarded Approaches

<!-- pending -->

## Risks & Open Threads

<!-- pending -->

## Build Plan

<!-- pending -->

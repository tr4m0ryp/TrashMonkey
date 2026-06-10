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

<!-- F# entries logged as they land -->

## References

<!-- R# entries -->

## Discarded Approaches

<!-- pending -->

## Risks & Open Threads

<!-- pending -->

## Build Plan

<!-- pending -->

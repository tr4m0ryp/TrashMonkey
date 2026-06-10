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

<!-- T1..T10 pending -->

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

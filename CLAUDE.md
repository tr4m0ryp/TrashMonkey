# yolo-waste-sorter

Fine-tune YOLO11n (Ultralytics) for six-class waste detection -- plastic, paper,
cardboard, metal, glass, organic -- trained on remapped public datasets only,
white-background presentation, exported to TensorRT FP16 on a Jetson Orin Nano.
"rest" is NOT a trained class: it is a confidence-threshold rule in control logic.

## Language & stack

Python >=3.11, `ultralytics` package. Pinned deps in `pyproject.toml`.
Config over hardcoding: hyperparameters and paths live in `configs/*.yaml`,
seeds via `src/yolo_waste_sorter/utils/seed.py` (seed 42 everywhere).

## Layout

- `data/{raw,interim,processed,external}/` -- gitignored; never mutate `raw/`
- `notebooks/` -- exploration ONLY, numbered (`01-explore-data.ipynb`)
- `src/yolo_waste_sorter/{data,features,models,visualization,utils}/` -- real logic
- `configs/` -- YAML experiment configs
- `experiments/`, `models/` -- run logs and checkpoints (gitignored)
- `reports/figures/` -- generated figures
- `paper.tex` + `refs.bib` -- the technical report (NeurIPS template, project root)
- `notes/`, `research/`, `tasks/` -- /flow pipeline docs (notes+tasks gitignored)

## Commands

- `make paper` -- build the report (tectonic)
- `make lint` / `make test`
- `make repro` -- data pipeline placeholder (download -> remap -> merge -> balance -> train -> export)

## Pipeline state (/flow)

Vision doc: `notes/yolo11-waste-detection-finetune.md` (slug: `yolo11-waste-detection-finetune`).
Next: `/flow:research yolo11-waste-detection-finetune`, then `/flow:refine`, then `/flow:readyforlaunch`.

## Documentation discipline (paper scribe is ACTIVE here)

`./paper.tex` exists, so the session-end scribe worker integrates findings into
the paper automatically. While working in this repo, log paper-worthy facts AS
THEY OCCUR -- confirmed results with numbers, settled design decisions,
limitations, datasets/tools worth citing:

```bash
bash ~/.claude/hooks/paper/log-finding.sh --title "..." --body "..." \
  --kind result|decision|caveat|method|citation --agent "main"
```

Force an immediate integrate: `FORCE=1 bash ~/.claude/hooks/paper/scribe.sh --root "$PWD"`.
Research rigor expected throughout: fixed seeds, pinned deps, out-of-sample eval,
experiment tracking, every reported number tied to a run config.

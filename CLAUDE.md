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
- `notebooks/` -- manager + exploration notebooks; see "Notebook discipline"
- `src/yolo_waste_sorter/{data,features,models,visualization,utils}/` -- real logic
- `configs/` -- YAML experiment configs
- `experiments/`, `models/` -- run logs and checkpoints (gitignored)
- `reports/figures/` -- generated figures
- `paper.tex` + `refs.bib` -- the technical report (NeurIPS template, project root)
- `notes/`, `research/`, `tasks/` -- /flow pipeline docs (notes+tasks gitignored)

## Notebook discipline (manager-notebook architecture)

Notebooks are MANAGERS, never code -- the ESPResso-V2 pattern
(`github.com/tr4m0ryp/ESPResso-V2`; full rules in global memory
`feedback/workflow/manager-notebooks.md`). Claude Code iterates on `.py` files
far more effectively than on notebooks, so:

- ALL logic lives in `src/yolo_waste_sorter/`; `notebooks/manager.ipynb` only
  imports and calls it. Zero model/data-transform logic in any notebook.
- Manager cell sequence: init + preflight validation -> repo clone/update
  (Colab) -> data cache check -> smoke test -> checkpoint detection/resume ->
  prepare data -> train (canary viability check, then full) -> evaluate ->
  plots -> summary.
- Must survive "Runtime > Run all" on a fresh Colab session: idempotent cells,
  `subprocess.run(..., check=True)` over bare `!cmd`, validate package
  importability instead of pinning Colab packages, device-agnostic torch.
- When a `src/` signature changes, update the notebook call site in the same
  edit (and vice versa).
- Training scripts support `SMOKE_TEST=1`/`--smoke` (tiny CPU run through the
  full cycle); every run appends to `experiments/runs.jsonl` (config, git
  hash, metrics, runtime); checkpoints + plots persist to Drive on Colab.
- Exploration notebooks stay separate and numbered (`01-explore-data.ipynb`)
  and never feed the pipeline.

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

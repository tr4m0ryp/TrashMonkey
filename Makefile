# Prefer the project venv when present so make targets, agents, and the
# pre-push gate all run against the same interpreter and dependencies.
PY      := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PYTEST  := $(shell [ -x .venv/bin/pytest ] && echo .venv/bin/pytest || echo pytest)
RUFF    := $(shell [ -x .venv/bin/ruff ] && echo .venv/bin/ruff || echo ruff)
MYPY    := $(shell [ -x .venv/bin/mypy ] && echo .venv/bin/mypy || echo mypy)

export NO_ALBUMENTATIONS_UPDATE = 1

.PHONY: paper lint test repro

paper:
	tectonic paper.tex

lint:
	$(RUFF) check src tests
	$(MYPY) src

test:
	$(PYTEST)

repro:
	PYTHONPATH=src $(PY) -m yolo_waste_sorter.data.pipeline run

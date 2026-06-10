.PHONY: paper lint test repro

paper:
	tectonic paper.tex

lint:
	ruff check src tests
	mypy src

test:
	pytest

repro:
	@echo "TODO: data pipeline (download -> remap -> merge -> balance -> train -> export)"

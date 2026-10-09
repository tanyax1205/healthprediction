# MultiDx developer shortcuts. Targets run from the repo root.
# Example:  make api     make test     make lint

.PHONY: api test lint train smoke

## Run the FastAPI server (reads .env; hot-reload in dev)
api:
	.venv/bin/python -m backend.run --reload

## Run all pytest suites (ml + backend)
test:
	.venv/bin/python -m pytest
	.venv/bin/python -m pytest backend/tests

## Lint everything with ruff
lint:
	.venv/bin/ruff check ml backend

## Retrain all 8 models via the CLI (never through the API)
train:
	.venv/bin/python -m ml.train --all

train-quick:
	.venv/bin/python -m ml.train --all --quick

## Quick end-to-end check against a running server
smoke:
	./scripts/smoke_api.sh
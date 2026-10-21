.DEFAULT_GOAL := help
PYTHON_VERSION = 3.13

## Create the virtual environment
.PHONY: venv
venv:
	uv venv --python $(PYTHON_VERSION)

## Install core + dev dependencies (add extras with: uv sync --extra open --extra data)
.PHONY: sync
sync:
	uv sync --locked

## Lint with ruff
.PHONY: lint
lint:
	uv run ruff check .

## Format with ruff
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

## Run tests
.PHONY: test
test:
	uv run pytest

## Serve the project website at http://localhost:8000
.PHONY: website
website:
	uv run python -m http.server 8000 --bind 127.0.0.1

## Show available targets
.PHONY: help
help:
	@awk '/^## /{desc=substr($$0,4); next} /^[a-z-]+:/{if (desc) {split($$1, t, ":"); printf "%-10s %s\n", t[1], desc}; desc=""}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
PYTHON_VERSION = 3.13

## Create the virtual environment
.PHONY: venv
venv:
	uv venv --python $(PYTHON_VERSION)

## Install core + dev dependencies (add extras with: uv sync --locked --extra open --extra data)
.PHONY: sync
sync:
	uv sync --locked $(EXTRAS)

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

## Data step 1: filter an OSM extract per NACE section (PBF=europe-latest.osm.pbf; needs osmium-tool)
.PHONY: data-filter
data-filter:
	uv run python -m moneta.preprocess.osm_filter --input $(PBF)

## Data step 2: combine filtered objects (Gold: name + address + external link)
.PHONY: data-extract
data-extract:
	uv run python -m moneta.preprocess.osm_extract --mode Gold

## Data step 3: sample 50 records per NACE section (seed 42)
.PHONY: data-sample
data-sample:
	uv run python -m moneta.preprocess.sample

## Data step 4: fetch Wikidata/Wikipedia/website text (optional LIMIT=N, OVERWRITE=1)
.PHONY: data-enrich
data-enrich:
	uv run python -m moneta.preprocess.enrich $(if $(LIMIT),--limit $(LIMIT)) $(if $(OVERWRITE),--overwrite)

## Data step 5: download OSM + satellite images (optional LIMIT=N)
.PHONY: data-images
data-images:
	uv run python -m moneta.preprocess.map_download $(if $(LIMIT),--limit $(LIMIT))

## Download the paper's enriched data and images (not yet hosted)
.PHONY: data-download
data-download:
	@echo "TODO: hosted MONETA_extended + images not available yet; use data-enrich / data-images" && exit 1

## Check tracked files for keys / machine paths and data/ for changes (run before pushing)
.PHONY: release-check
release-check:
	./scripts/release_check.sh

## Serve the project website at http://localhost:8000
.PHONY: website
website:
	uv run python -m http.server 8000 --bind 127.0.0.1

## Show available targets
.PHONY: help
help:
	@awk '/^## /{desc=substr($$0,4); next} /^[a-z-]+:/{if (desc) {split($$1, t, ":"); printf "%-10s %s\n", t[1], desc}; desc=""}' $(MAKEFILE_LIST)

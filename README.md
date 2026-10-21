# MONETA

Code and data for **MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems** (ACL 2026).
[Paper](https://aclanthology.org/2026.acl-long.674/) · [arXiv](https://arxiv.org/abs/2604.07956) · [Project page](https://trusthlt.github.io/Moneta/)

MONETA is the first multimodal benchmark for industry classification: 1,000 European businesses from OpenStreetMap, labelled with 20 NACE sections, with text (website, Wikipedia, Wikidata) and geospatial (OSM map, satellite image) sources.

## Repository

```
data/Moneta/        MONETA.json (1,000 records), MONETA_10K.jsonl
data/filter/        NACE descriptions and the NACE -> OSM tag mapping
configs/config.yaml paths, model ids (pinned HF revisions), data construction settings
moneta/             package: dataloader, prompts, models, pipelines, evaluation, preprocess
tests/              unit tests (no GPU, network or API keys needed)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync --locked --extra open --extra data   # open: torch + transformers, data: data construction
make test                                    # optional
```

`flash-attn` is optional; without it Qwen2.5-VL uses SDPA attention, which can change a few predictions slightly.
To use local model weights instead of the Hugging Face Hub, create `configs/local.yaml` (gitignored):

```yaml
models:
  InternVL3-8B: {path: /path/to/InternVL3-8B, revision: null}
```

## Data

`data/Moneta/MONETA.json` holds the OSM records (id, type, tags, bbox, NACE `category`). Experiments with
text sources or images additionally need the fetched source text and the OSM / satellite images:

```bash
make data-enrich    # Wikidata / Wikipedia / website text -> data/processed/MONETA_extended.json
make data-images    # OSM + satellite images             -> data/processed/images/{osm,satellite}/
```

Both accept `LIMIT=N` for a quick test (enriched text then goes to `data/processed/limit<N>/`);
`make data-enrich OVERWRITE=1` replaces an existing enriched file. Web content changes over time, so
newly fetched text differs from the text used in the paper.

The dataset itself was built from an OSM Europe extract ([Geofabrik](https://download.geofabrik.de/europe.html)):

```bash
make data-filter PBF=europe-latest.osm.pbf   # needs osmium-tool, e.g. conda install -c conda-forge osmium-tool
make data-extract                            # Gold: name + address + external link
make data-sample                             # 50 per NACE section, seed 42 (= data/Moneta/MONETA.json)
```

MONETA-10K was sampled from the same Gold extract; its sampling is not scripted.

## Experiments

Open MLLMs: Qwen2.5-VL (7B, 32B, 72B), InternVL 2.5 (1B, 4B, 38B), InternVL 3 (8B, 14B, 38B, 78B); see `configs/config.yaml`.
Resources: `osm`, `satellite`, `source` (Wikidata, Wikipedia, website) or `none`. All commands take `--limit N`.

```bash
# Table 2: zero-shot baseline (Simple prompt, label only)
uv run python -m moneta.zero_shot --model_name InternVL3-8B --data_resources satellite osm source \
  --prompt_config Simple --output_format Text --batch_size 8

# Table 3: explanation (--output_format JSON) and extended prompt (--prompt_config Extended)
uv run python -m moneta.zero_shot --model_name InternVL3-8B --data_resources satellite osm source \
  --prompt_config Extended --output_format JSON

# Table 3: multi-turn = one clue per source, then a final decision from all clues
uv run python -m moneta.clue_generator --model_name InternVL3-8B
uv run python -m moneta.multiturn --model_name InternVL3-8B --prompt_config Extended --output_format JSON  # mixture
```

Results go to `results/<run>/inference_results.json` (one record per business with `llm_response`, and
`llm_explanation` for JSON output) plus `run_config.json` (model path, revision, data path).
`--limit N` runs are written to `results/limit<N>/` (evaluate them with `--results results/limit<N>`);
clues go to `results/clues/[limit<N>/]<model>/clues.json`.

## Evaluation

```bash
uv run python -m moneta.evaluate --results results            # accuracy, macro P/R/F1, unknown ratio -> evaluation_results.csv
uv run python -m moneta.clue_evaluate --input_dir results     # clue IDR / correctness / effectiveness of MultiTurn runs
```

Closed models (GPT-5, Gemini 2.5) and the fine-tuning comparison are not part of this release yet.

## Development

`make lint`, `make format`, `make test`; `make release-check` scans tracked files for keys and internal paths before a push.

## Abstract

Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10% and 74.10% with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80% with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines.

## Cite
```bibtex
@inproceedings{yuksel-etal-2026-moneta,
    title = "{MONETA}: Multimodal Industry Classification through Geographic Information with Multi Agent Systems",
    author = {Y{\"u}ksel, Arda  and
      Thiem, Gabriel  and
      Walter, Susanne  and
      Felka, Patrick  and
      Werb, Gabriela Alves  and
      Habernal, Ivan},
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.674/",
    pages = "14790--14814",
    ISBN = "979-8-89176-390-6",
    abstract = "Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10{\%} and 74.10{\%} with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80{\%} with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines."
}
```

## License

MIT, see [LICENSE](LICENSE).

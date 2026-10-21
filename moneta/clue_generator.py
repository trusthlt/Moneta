"""
Multi-turn step 1: one clue per modality (osm, satellite, wikidata, wikipedia, website) and item.
Writes <output_dir>/<model>/clues.json (dumped items with history.<modality>_clues); input of multiturn.py.
"""

import argparse
from itertools import batched

from tqdm import tqdm

from moneta.dataloader.moneta import MonetaDataset
from moneta.models import get_model
from moneta.prompts.multi_turn import MODALITY_PROMPT_MAP
from moneta.schema.experiment import ModelConfigSchema
from moneta.utils import get_logger, output_root, write_json

logger = get_logger()
MODALITIES = ["osm", "satellite", "wikidata", "wikipedia", "website"]


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--data_path", type=str, default=None, help="Default: paths.data_extended"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Default: paths.clues"
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def process_modality(modality, batch, model, prompt):
    """Clues for one modality; items without that input get None."""
    results = [None] * len(batch)
    if modality in ("osm", "satellite"):
        inputs = [
            item.image_paths.model_dump().get(f"{modality}_path") for item in batch
        ]
    else:
        inputs = [item.sources.model_dump().get(modality) for item in batch]
    indexes = [i for i, x in enumerate(inputs) if x is not None]
    if not indexes:
        return results
    present = [inputs[i] for i in indexes]
    if modality in ("osm", "satellite"):
        clues = model.run(present, None, prompt)
    else:
        clues = model.run(None, present, prompt)
    for idx, clue in zip(indexes, clues, strict=True):
        results[idx] = clue
    return results


def main(
    model_settings: ModelConfigSchema, data_path=None, output_dir=None, limit=None
):
    dataset = MonetaDataset(
        ["satellite", "osm", "source"], data_path=data_path, limit=limit
    )
    logger.info(f"Loaded {len(dataset)} items from {dataset.data_path}")
    model = get_model(model_settings)
    results = []
    for batch in tqdm(
        list(batched(dataset.data, model_settings.batch_size, strict=False)),
        desc="Batches",
    ):
        clues = {
            m: process_modality(m, batch, model, MODALITY_PROMPT_MAP[m])
            for m in MODALITIES
        }
        for index, item in enumerate(batch):
            result = item.model_dump()
            result["history"] = {f"{m}_clues": clues[m][index] for m in MODALITIES}
            results.append(result)
    folder = output_root(output_dir, "clues", limit) / model_settings.model_name
    path = folder / "clues.json"
    write_json(results, path)
    run_info = {
        "model": getattr(model, "source", None),
        "data_path": str(dataset.data_path),
        "limit": limit,
    }
    write_json(
        {"model_settings": model_settings.model_dump(), **run_info},
        folder / "run_config.json",
    )
    logger.info(f"Saved clues for {len(results)} items to {path}")
    return path


if __name__ == "__main__":
    args = get_parser().parse_args()
    settings = ModelConfigSchema(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    main(
        settings, data_path=args.data_path, output_dir=args.output_dir, limit=args.limit
    )

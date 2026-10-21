"""
Multi-turn step 2: classify from the clues of all modalities plus the name (Table 3 multi-turn / mixture).
Reads clues.json from clue_generator.py; writes
<output_dir>/MultiTurn_NACE_<prompt>_<format>_<model>_osm_satellite_source/inference_results.json
"""

import argparse
from itertools import batched

from tqdm import tqdm

from moneta.dataloader.moneta import MonetaDataset
from moneta.models import get_model
from moneta.prompts.builder import generate_multi_turn_prompt
from moneta.schema.experiment import ExperimentConfigSchema, ModelConfigSchema
from moneta.utils import get_logger, load_config, resolve_path, save_results, to_result

logger = get_logger()


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--prompt_config", type=str, default="Simple", choices=["Simple", "Extended"]
    )
    parser.add_argument(
        "--output_format", type=str, default="Text", choices=["JSON", "Text"]
    )
    parser.add_argument(
        "--clue_path",
        type=str,
        default=None,
        help="Default: paths.clues/<model>/clues.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Default: paths.results"
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def generate_clue_texts(batch):
    """'<modality>_clues: <clue>' lines for all available clues, then the name."""
    texts = []
    for b in batch:
        text = ""
        for key, value in b.history.model_dump().items():
            if value and value != "cuda OOM":
                text += f"{key}: {value}\n"
        text += f"Name: {b.name}\n"
        texts.append(text)
    return texts


def main(
    exp_config: ExperimentConfigSchema, clue_path=None, output_dir=None, limit=None
):
    settings = exp_config.model_settings
    clue_path = (
        clue_path
        or resolve_path(load_config()["paths"]["clues"])
        / settings.model_name
        / "clues.json"
    )
    dataset = MonetaDataset([], data_path=clue_path, limit=limit)
    logger.info(f"Loaded clues for {len(dataset)} items from {dataset.data_path}")
    model = get_model(settings)
    system_prompt = generate_multi_turn_prompt(
        output_format=exp_config.output_format, prompt_config=exp_config.prompt_config
    )
    results = []
    for batch in tqdm(
        list(batched(dataset.data, settings.batch_size, strict=False)), desc="Batches"
    ):
        responses = model.run(None, generate_clue_texts(batch), system_prompt)
        results += [
            to_result(item, r, exp_config.output_format)
            for item, r in zip(batch, responses, strict=True)
        ]
    run_info = {
        "model": getattr(model, "source", None),
        "data_path": str(dataset.data_path),
        "limit": limit,
    }
    return save_results(results, "MultiTurn", exp_config, run_info, output_dir, limit)


if __name__ == "__main__":
    args = get_parser().parse_args()
    exp_config = ExperimentConfigSchema(
        model_settings=ModelConfigSchema(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        prompt_config=args.prompt_config,
        output_format=args.output_format,
        data_resources=[
            "osm",
            "satellite",
            "source",
        ],  # multi-turn always uses all clues
    )
    logger.info(exp_config.model_dump_json())
    main(
        exp_config,
        clue_path=args.clue_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )

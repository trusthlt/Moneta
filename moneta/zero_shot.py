"""
Zero-shot NACE classification with open MLLMs (Table 2; Table 3 explanation / extended prompt).
Writes <output_dir>/ZeroShot_NACE_<prompt>_<format>_<model>_<resources>/inference_results.json
"""

import argparse
from itertools import batched

from tqdm import tqdm

from moneta.dataloader.moneta import MonetaDataset
from moneta.models import get_model
from moneta.prompts.builder import generate_zero_shot_prompt
from moneta.schema.experiment import ExperimentConfigSchema, ModelConfigSchema
from moneta.utils import get_logger, save_results, to_result

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
        "--output_format", type=str, default="JSON", choices=["JSON", "Text"]
    )
    parser.add_argument(
        "--data_resources",
        nargs="+",
        default=["osm", "satellite"],
        help="osm satellite source, or none",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Default: paths.data / paths.data_extended",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Default: paths.results"
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def load_batch_data(batch):
    """Per item: a dict of image paths (osm / satellite) and the text context (sources + name)."""
    images, texts = [], []
    for b in batch:
        image_dict = {}
        if b.image_paths:
            if b.image_paths.osm_path:
                image_dict["osm"] = b.image_paths.osm_path
            if b.image_paths.satellite_path:
                image_dict["satellite"] = b.image_paths.satellite_path
        if image_dict:
            images.append(image_dict)

        context = ""
        if b.sources:
            if b.sources.wikidata:
                context += f"Wikidata: {b.sources.wikidata}\n"
            if b.sources.wikipedia:
                context += f"Wikipedia: {b.sources.wikipedia}\n"
            if b.sources.website:
                context += f"Website: {b.sources.website}\n"
        if b.name:
            context += f"Name: {b.name}\n"
        if context:
            texts.append(context.strip())
    return images or None, texts


def main(
    exp_config: ExperimentConfigSchema, data_path=None, output_dir=None, limit=None
):
    settings = exp_config.model_settings
    dataset = MonetaDataset(exp_config.data_resources, data_path=data_path, limit=limit)
    logger.info(f"Loaded {len(dataset)} items from {dataset.data_path}")
    model = get_model(settings)
    system_prompt = generate_zero_shot_prompt(
        output_format=exp_config.output_format,
        prompt_config=exp_config.prompt_config,
        resources=exp_config.data_resources,
    )
    results = []
    for batch in tqdm(
        list(batched(dataset.data, settings.batch_size, strict=False)), desc="Batches"
    ):
        images, texts = load_batch_data(batch)
        responses = model.run(images, texts, system_prompt)
        results += [
            to_result(item, r, exp_config.output_format)
            for item, r in zip(batch, responses, strict=True)
        ]
    run_info = {
        "model": getattr(model, "source", None),
        "data_path": str(dataset.data_path),
        "limit": limit,
    }
    return save_results(results, "ZeroShot", exp_config, run_info, output_dir, limit)


if __name__ == "__main__":
    args = get_parser().parse_args()
    resources = [] if args.data_resources == ["none"] else args.data_resources
    exp_config = ExperimentConfigSchema(
        model_settings=ModelConfigSchema(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        prompt_config=args.prompt_config,
        output_format=args.output_format,
        data_resources=resources,
    )
    logger.info(exp_config.model_dump_json())
    main(
        exp_config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )

"""
utils.py
Cross-cutting helpers: config, paths, JSON I/O, logging, run-folder naming and
LLM JSON-response parsing.
"""

import json
from pathlib import Path
import re
import sys

from loguru import logger
import yaml

from moneta.schema.experiment import ExperimentConfigSchema


def get_logger():
    logger.remove()
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    logger.add(sys.stdout, level="INFO", format=format_str)
    return logger


def root_path() -> Path:
    return Path(__file__).parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else root_path() / path


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


def read_json_list(path, lines: bool | None = None) -> list:
    """Read a JSON array or a JSONL file (inferred from the .jsonl suffix if lines is None)."""
    if lines is None:
        lines = str(path).endswith(".jsonl")
    with open(path, encoding="utf-8") as f:
        if lines:
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def write_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def model_name_to_path(model_name: str, config: dict | None = None) -> str:
    """Map a model name (as used in run names) to a HF Hub id or local directory (configs/config.yaml: models)."""
    models = (config or load_config())["models"]
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Known: {sorted(models)}")
    return models[model_name]


def repair_json_string_end(malformed_json_str):
    """
    Attempt to repair the end of a malformed JSON string.
    If it ends with ```, assume it's complete.
    If it ends with "}, assume it's missing the closing ```
    If it ends with ", assume it's missing the closing " and } and ```
    """
    malformed_json_str = malformed_json_str.strip()
    if malformed_json_str.endswith("```"):
        return malformed_json_str
    elif malformed_json_str.endswith('"'):
        return malformed_json_str + "\n}\n```"
    elif malformed_json_str.endswith("}"):
        return malformed_json_str + "\n```"
    else:
        # Assuming it ended mid-string, try to close the string and object
        return malformed_json_str + '"\n}\n```'


def parse_json_response(response_text):
    """
    Parse a JSON response from the LLM.
    Response should contain a JSON with keys "LLM_RESPONSE" and optionally "EXPLANATION".
    Returns a tuple (LLM_RESPONSE, EXPLANATION); ("parse_err", response_text) if parsing fails.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        modified_response = repair_json_string_end(response_text)
        match = re.search(r"\{.*\}", modified_response, re.DOTALL)

    if not match:
        return ("parse_err", response_text)

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ("parse_err", response_text)
    return parsed.get("LLM_RESPONSE", "UNKNOWN"), parsed.get("EXPLANATION", "")


def generate_folder_name(pipeline: str, exp_config: ExperimentConfigSchema):
    """
    Generate a run-folder name from the experiment configuration.
    e.g. ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_osm_satellite_source
    The pipeline name must not contain '_' (parse_folder_name splits on it).
    """
    if "_" in pipeline:
        raise ValueError(f"Pipeline name must not contain '_': {pipeline}")
    standardized_model_name = (
        exp_config.model_settings.model_name.replace("_", "-")
        .replace(" ", "-")
        .replace(".", "-")
    )
    folder_name = (
        f"{pipeline}_{exp_config.task.value}_"
        f"{exp_config.prompt_config.value}_{exp_config.output_format.value}_"
        f"{standardized_model_name}_{'_'.join(exp_config.data_resources)}"
    )
    if folder_name.endswith("_"):
        folder_name = folder_name + "none"
    return folder_name


def parse_folder_name(folder_name: str):
    """
    Parse a run-folder name back into its experiment attributes.
    e.g. ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_osm_satellite_source
    Model names come back with '-' replaced by ' '; 'none' gives no data resources.
    Raises ValueError if the folder name is malformed.
    """
    parts = folder_name.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Folder name {folder_name} does not have enough parts to parse."
        )
    data_resources = parts[5:] if len(parts) > 5 else ["none"]
    if data_resources == ["none"]:
        data_resources = []
    return {
        "pipeline": parts[0],
        "task": parts[1],
        "prompt_config": parts[2],
        "output_format": parts[3],
        "model_name": parts[4].replace("-", " "),
        "data_resources": data_resources,
    }

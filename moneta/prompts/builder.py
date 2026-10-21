"""
System prompts for zero-shot and multi-turn NACE classification.
Simple: section codes and titles. Extended: adds descriptions, inclusions and exclusions.
"""

from moneta.prompts.multi_turn import FINAL_PROMPT_NACE
from moneta.prompts.zero_shot import (
    BASELINE_PROMPT_NACE,
    JSON_OUTPUT_PROMPT,
    TEXT_OUTPUT_PROMPT,
)
from moneta.utils import load_config, read_json_list, resolve_path


def _get_nace_content(prompt_config: str = "Simple") -> str:
    nace_json = read_json_list(
        resolve_path(load_config()["paths"]["nace_descriptions"])
    )
    prompt = ""
    for item in nace_json:
        current_prompt = f"{item['category']}: {item['title']}\n"
        if prompt_config == "Extended":
            if item["description"] != "":
                current_prompt += f"Description: {item['description']}\n"
            if item["inclusion"] != "":
                current_prompt += f"Included: {item['inclusion']}\n"
            if item["exclusion"] != "":
                current_prompt += f"Excluded: {item['exclusion']}\n"
        prompt += current_prompt
    return prompt.rstrip()


def _get_output_format(output_format: str = "Text") -> str:
    return JSON_OUTPUT_PROMPT if output_format == "JSON" else TEXT_OUTPUT_PROMPT


def generate_zero_shot_prompt(
    output_format: str = "Text",
    prompt_config: str = "Simple",
    resources: list | None = None,
) -> str:
    resource_prompt = ",".join(resources) if resources else "None"
    return BASELINE_PROMPT_NACE.format(
        NACE_CONTEXT=_get_nace_content(prompt_config),
        OUTPUT_FORMAT=_get_output_format(output_format),
        resources=resource_prompt,
    ).strip()


def generate_multi_turn_prompt(
    output_format: str = "Text", prompt_config: str = "Simple"
) -> str:
    return FINAL_PROMPT_NACE.format(
        NACE_CONTEXT=_get_nace_content(prompt_config),
        OUTPUT_FORMAT=_get_output_format(output_format),
    ).strip()

from pathlib import Path

import pytest

from moneta.prompts.builder import generate_multi_turn_prompt, generate_zero_shot_prompt

# Golden prompts rendered by the original research code
GOLDEN = Path(__file__).parent / "data" / "prompts"


@pytest.mark.parametrize(
    "name, prompt",
    [
        (
            "zero_shot_Simple_Text_osm-satellite-source",
            lambda: generate_zero_shot_prompt(
                "Text", "Simple", ["osm", "satellite", "source"]
            ),
        ),
        (
            "zero_shot_Extended_JSON_none",
            lambda: generate_zero_shot_prompt("JSON", "Extended", []),
        ),
        (
            "multi_turn_Simple_Text",
            lambda: generate_multi_turn_prompt("Text", "Simple"),
        ),
        (
            "multi_turn_Extended_JSON",
            lambda: generate_multi_turn_prompt("JSON", "Extended"),
        ),
    ],
)
def test_prompts_match_original(name, prompt):
    assert prompt() == (GOLDEN / f"{name}.txt").read_text()


def test_extended_adds_descriptions():
    simple = generate_zero_shot_prompt("Text", "Simple", ["satellite"])
    extended = generate_zero_shot_prompt("Text", "Extended", ["satellite"])
    assert "Description:" not in simple
    assert "Description:" in extended
    assert "Following sources will be provided: satellite" in simple


def test_output_formats():
    assert "LLM_RESPONSE" in generate_zero_shot_prompt("JSON", "Simple", [])
    assert "SINGLE TOKEN RESPONSE ONLY" in generate_zero_shot_prompt(
        "Text", "Simple", []
    )
    assert "Following sources will be provided: None" in generate_zero_shot_prompt(
        "Text", "Simple", []
    )

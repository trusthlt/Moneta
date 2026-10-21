import json

import pytest

from moneta.schema.experiment import (
    ExperimentConfigSchema,
    ModelConfigSchema,
    OutputFormat,
    PromptTemplate,
    TaskType,
)
from moneta.utils import (
    generate_folder_name,
    model_name_to_path,
    parse_folder_name,
    parse_json_response,
    read_json_list,
    repair_json_string_end,
)


def unfence(text):
    return text.strip().removeprefix("```json").removesuffix("```").strip()


@pytest.fixture
def model_config():
    return ModelConfigSchema(
        model_name="InternVL2_5-38B", batch_size=4, max_length=1024
    )


@pytest.fixture
def experiment_config(model_config):
    return ExperimentConfigSchema(
        model_settings=model_config,
        task=TaskType.NACE,
        prompt_config=PromptTemplate.SIMPLE,
        output_format=OutputFormat.JSON,
        data_resources=["osm", "satellite", "source"],
    )


@pytest.fixture
def correct_json_string():
    return """```json
    {
        "key": "value"
    }
    ```"""


@pytest.fixture
def malformed_json_string():
    return '''```json
    {
        "key": "value"'''


def test_generate_folder_name(experiment_config):
    folder_name = generate_folder_name("ZeroShot", experiment_config)
    expected_name = "ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_osm_satellite_source"
    assert folder_name == expected_name


def test_generate_folder_name_no_data_resources(experiment_config):
    experiment_config.data_resources = []
    folder_name = generate_folder_name("ZeroShot", experiment_config)
    expected_name = "ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_none"
    assert folder_name == expected_name


def test_parse_folder_name():
    folder_name = "ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_osm_satellite_source"
    folder_attributes = parse_folder_name(folder_name)

    assert folder_attributes["pipeline"] == "ZeroShot"
    assert folder_attributes["task"] == "NACE"
    assert folder_attributes["prompt_config"] == "Simple"
    assert folder_attributes["output_format"] == "JSON"
    assert (
        folder_attributes["model_name"] == "InternVL2 5 38B"
    )  # Note the replacement of '-' with ' '
    assert folder_attributes["data_resources"] == ["osm", "satellite", "source"]


def test_parse_folder_name_no_data_resources():
    folder_name = "ZeroShot_NACE_Simple_JSON_InternVL2-5-38B_none"
    folder_attributes = parse_folder_name(folder_name)
    assert folder_attributes["pipeline"] == "ZeroShot"
    assert folder_attributes["task"] == "NACE"
    assert folder_attributes["prompt_config"] == "Simple"
    assert folder_attributes["output_format"] == "JSON"
    assert (
        folder_attributes["model_name"] == "InternVL2 5 38B"
    )  # Note the replacement of '-' with ' '
    assert folder_attributes["data_resources"] == []


def test_repair_json_string_end_complete(correct_json_string):
    repaired = repair_json_string_end(correct_json_string)
    json_repaired = json.loads(unfence(repaired))
    json_original = json.loads(unfence(correct_json_string))
    assert json_repaired == json_original


def test_repair_json_string_end_missing_closing_brace(malformed_json_string):
    repaired = repair_json_string_end(malformed_json_string)
    expected = """```json
    {
        "key": "value"
    }
    ```"""
    json_expected = json.loads(unfence(expected))
    json_repaired = json.loads(unfence(repaired))
    assert json_expected == json_repaired


def test_generate_folder_name_rejects_underscore_pipeline(experiment_config):
    with pytest.raises(ValueError):
        generate_folder_name("zero_shot", experiment_config)


def test_folder_name_round_trip(experiment_config):
    parsed = parse_folder_name(generate_folder_name("MultiTurn", experiment_config))
    assert parsed["pipeline"] == "MultiTurn"
    assert parsed["data_resources"] == ["osm", "satellite", "source"]


@pytest.mark.parametrize(
    "response, expected",
    [
        (
            '```json\n{"LLM_RESPONSE": "C", "EXPLANATION": "factory"}\n```',
            ("C", "factory"),
        ),
        ('{"LLM_RESPONSE": "G"}', ("G", "")),
        ('```json\n{"LLM_RESPONSE": "I", "EXPLANATION": "hotel', ("I", "hotel")),
        ("I think it is C", ("parse_err", "I think it is C")),
    ],
)
def test_parse_json_response(response, expected):
    assert parse_json_response(response) == expected


def test_read_json_list_infers_jsonl(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": 1}\n{"id": 2}\n')
    (tmp_path / "a.json").write_text('[{"id": 1}]')
    assert read_json_list(tmp_path / "a.jsonl") == [{"id": 1}, {"id": 2}]
    assert read_json_list(tmp_path / "a.json") == [{"id": 1}]


def test_model_name_to_path():
    config = {"models": {"InternVL3-8B": "OpenGVLab/InternVL3-8B"}}
    assert model_name_to_path("InternVL3-8B", config) == "OpenGVLab/InternVL3-8B"
    with pytest.raises(ValueError):
        model_name_to_path("Llava-v1.6-7B", config)

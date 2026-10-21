from pydantic import ValidationError
import pytest

from moneta.schema.experiment import (
    ExperimentConfigSchema,
    ModelConfigSchema,
    OutputFormat,
    PromptTemplate,
    TaskType,
)


def test_model_config_schema_defaults():
    m = ModelConfigSchema()
    assert m.model_name == "Qwen2.5-VL-7B"
    assert m.batch_size == 1
    assert m.max_length == 512


def test_model_config_schema_values():
    m = ModelConfigSchema(
        model_name="test",
        batch_size=8,
        max_length=1024,
    )
    assert m.model_name == "test"
    assert m.batch_size == 8
    assert m.max_length == 1024


def test_model_config_schema_invalid_batch_size():
    with pytest.raises(ValidationError):
        ModelConfigSchema(batch_size=0)


def test_model_config_schema_invalid_max_length():
    with pytest.raises(ValidationError):
        ModelConfigSchema(max_length=5000)


def test_experiment_config_schema_defaults():
    e = ExperimentConfigSchema()
    assert isinstance(e.model_settings, ModelConfigSchema)
    assert e.task == "NACE"
    assert e.prompt_config == "Simple"
    assert e.data_resources == ["osm", "satellite"]


def test_experiment_config_schema_values():
    m = ModelConfigSchema(
        model_name="test",
        batch_size=8,
        max_length=1024,
    )
    e = ExperimentConfigSchema(
        model_settings=m,
        task=TaskType.WEBSITE,
        prompt_config=PromptTemplate.EXTENDED,
        output_format=OutputFormat.TEXT,
        data_resources=["resource1", "resource2"],
    )
    assert e.model_settings == m
    assert e.task == "WEBSITE"
    assert e.prompt_config == "Extended"
    assert e.output_format == "Text"
    assert e.data_resources == ["resource1", "resource2"]


def test_experiment_config_schema_invalid_task():
    with pytest.raises(ValidationError):
        ExperimentConfigSchema(task="INVALID_TASK")


def test_experiment_config_schema_invalid_prompt_config():
    with pytest.raises(ValidationError):
        ExperimentConfigSchema(prompt_config="INVALID_PROMPT")


def test_experiment_config_schema_invalid_output_format():
    with pytest.raises(ValidationError):
        ExperimentConfigSchema(output_format="INVALID_FORMAT")

"""
Schemas for configuring experiments with different models and tasks.
PromptTemplate: Simple prompt contains category names; Extended adds category descriptions.
OutputFormat: JSON output carries an explanation, Text output only the label.
TaskType: NACE (MONETA); NAICS and WEBSITE are used by the fine-tuning experiments.
ModelConfigSchema: model_name, batch_size, max_length.
ExperimentConfigSchema: model_settings, task, prompt_config, output_format, data_resources.
"""

from enum import Enum

from pydantic import BaseModel, Field


class PromptTemplate(str, Enum):
    SIMPLE = "Simple"
    EXTENDED = "Extended"


class OutputFormat(str, Enum):
    JSON = "JSON"
    TEXT = "Text"


class TaskType(str, Enum):
    NACE = "NACE"
    NAICS = "NAICS"
    WEBSITE = "WEBSITE"


class ModelConfigSchema(BaseModel):
    model_name: str = Field(default="Qwen2.5-VL-7B")
    batch_size: int = Field(default=1, ge=1, le=256)
    max_length: int = Field(default=512, ge=1, le=2048)


class ExperimentConfigSchema(BaseModel):
    model_settings: ModelConfigSchema = Field(default_factory=ModelConfigSchema)
    task: TaskType = Field(default=TaskType.NACE)
    prompt_config: PromptTemplate = Field(default=PromptTemplate.SIMPLE)
    output_format: OutputFormat = Field(default=OutputFormat.JSON)
    data_resources: list[str] = ["osm", "satellite"]

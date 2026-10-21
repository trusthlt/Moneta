from moneta.schema.experiment import ModelConfigSchema


def get_model(model_settings: ModelConfigSchema):
    """InternVL 2.5/3 or Qwen2.5-VL, chosen by model name (imported lazily: torch is an optional extra)."""
    name = model_settings.model_name.lower()
    if "intern" in name:
        from moneta.models.internvl import InternVL

        return InternVL(model_settings)
    if "qwen" in name:
        from moneta.models.qwen import QwenVL

        return QwenVL(model_settings)
    raise ValueError(f"Model {model_settings.model_name} is not supported.")

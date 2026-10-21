"""Base class for the open MLLMs: settings, GPU memory cleanup and CUDA OOM handling."""

import gc

import torch

from moneta.schema.experiment import ModelConfigSchema
from moneta.utils import get_logger

logger = get_logger()


class BaseMLLM:
    def __init__(self, model_properties: ModelConfigSchema):
        self.name = model_properties.model_name
        self.batch_size = model_properties.batch_size
        self.max_length = model_properties.max_length

    @staticmethod
    def empty_memory():
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def handle_oom(size: int) -> list[str]:
        """Skip the batch: every item gets the response 'cuda OOM'."""
        logger.warning("CUDA OOM, skipping batch and releasing memory")
        BaseMLLM.empty_memory()
        return ["cuda OOM"] * size

    def run(self, images: list | None, texts: list | None, prompt: str) -> list[str]:
        """
        images: per item a path or a dict of paths ({'osm': ..., 'satellite': ...}), or None.
        texts: per item a string, or None. prompt: system prompt.
        Returns one decoded response per item.
        """
        raise NotImplementedError

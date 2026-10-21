"""Qwen2.5-VL via HF transformers (batched, left padding)."""

import importlib.util

from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from moneta.models.base import BaseMLLM
from moneta.schema.experiment import ModelConfigSchema
from moneta.utils import get_logger, model_source

logger = get_logger()


class QwenVL(BaseMLLM):
    def __init__(self, model_properties: ModelConfigSchema):
        super().__init__(model_properties)
        path, revision = model_source(model_properties.model_name)
        # The paper runs used flash-attention 2; SDPA is the fallback when flash-attn is not installed.
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        if attn == "sdpa":
            logger.warning(
                "flash-attn not installed: using SDPA, predictions may differ slightly"
            )
        self.source = {"path": path, "revision": revision, "attn_implementation": attn}
        logger.info(f"Loading {self.source}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            revision=revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoProcessor.from_pretrained(
            path,
            revision=revision,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.tokenizer.tokenizer.padding_side = "left"

    @staticmethod
    def _get_image_message(image):
        images = []
        if isinstance(image, dict):
            for key in image:
                images.append(Image.open(image[key]).convert("RGB"))
        elif isinstance(image, str):
            images.append(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            images.append(image.convert("RGB"))
        return [{"type": "image", "image": image} for image in images]

    @staticmethod
    def _get_text_message(text):
        if isinstance(text, dict):
            return [
                {"type": "text", "text": f"{key}: {value}"}
                for key, value in text.items()
            ]
        if isinstance(text, str):
            return [{"type": "text", "text": text}]
        if isinstance(text, list):
            return [{"type": "text", "text": t} for t in text]
        raise ValueError(f"Unsupported text format: {type(text)}")

    @staticmethod
    def get_message(image, text, prompt):
        """Text parts come before image parts in the user turn."""
        content = []
        if text:
            content += QwenVL._get_text_message(text)
        if image:
            content += QwenVL._get_image_message(image)
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    def run(
        self, images: list | None = None, texts: list | None = None, prompt: str = ""
    ) -> list[str]:
        if not images and not texts:
            raise ValueError("At least one of 'images' or 'texts' must be provided.")
        images = images or [None] * len(texts)
        texts = texts or [None] * len(images)
        messages = [
            self.get_message(image, text, prompt)
            for image, text in zip(images, texts, strict=True)
        ]
        try:
            text_inputs = [
                self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]
            image_inputs = (
                process_vision_info(messages)[0] if images[0] is not None else None
            )
            inputs = self.tokenizer(
                text=text_inputs, images=image_inputs, padding=True, return_tensors="pt"
            ).to("cuda")
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_length
            )
            trimmed = [
                out[len(inp) :]
                for inp, out in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            return self.tokenizer.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except torch.cuda.OutOfMemoryError:
            return self.handle_oom(len(images))
        finally:
            self.empty_memory()

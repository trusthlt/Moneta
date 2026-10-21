"""InternVL 2.5 / 3 via HF transformers remote code (batch_chat, greedy decoding).
Image preprocessing follows the official InternVL model cards; OSM and satellite
patches of one item are concatenated behind a single <image> placeholder."""

import math

from PIL import Image
import torch
import torchvision.transforms as T  # noqa: N812 (as in the InternVL model card)
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from moneta.models.base import BaseMLLM
from moneta.schema.experiment import ModelConfigSchema
from moneta.utils import model_source

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_images(image_input, input_size=448, max_num=12):
    """
    Accepts:
        - A single image path (str)
        - A dict of image paths (e.g., {"front": "a.jpg", "back": "b.jpg"})
    Returns:
        A tensor of shape [N_patches_total, C, H, W]
    """

    def _load_and_patch(image_path):
        image = Image.open(image_path).convert("RGB")
        patches = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        transform = build_transform(input_size=input_size)
        return [transform(p) for p in patches]

    if isinstance(image_input, str):
        pixel_values = _load_and_patch(image_input)

    elif isinstance(image_input, dict):
        pixel_values = []
        for path in image_input.values():
            if path:
                pixel_values.extend(_load_and_patch(path))
        if not pixel_values:
            return None  # all paths were None/empty

    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input)}. Expected str or dict."
        )

    return torch.stack(pixel_values)


def prepare_text_input(text) -> str:
    if isinstance(text, str):
        return text
    elif isinstance(text, dict):
        text = "\n".join(f"{key}: {value}" for key, value in text.items())
        return text
    else:
        raise ValueError(
            f"Unsupported text format: {type(text)}. Expected str, dict, or list."
        )


def prepare_prompt(
    system_prompt: str, text_prompt: str = None, has_image: bool = False
) -> str:
    prompt = system_prompt
    if text_prompt:
        prompt += f"\n{text_prompt}\n"
    if has_image:
        prompt += "\n<image>\n"
    return prompt.strip()


class InternVL(BaseMLLM):
    def __init__(self, model_properties: ModelConfigSchema):
        super().__init__(model_properties)
        path, revision = model_source(model_properties.model_name)
        self.source = {"path": path, "revision": revision}
        self.model = AutoModel.from_pretrained(
            path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.split_model(self.name, path, revision),
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, revision=revision, trust_remote_code=True
        )

    @staticmethod
    def split_model(model_name, path, revision):
        """Spread LLM layers over all GPUs; GPU 0 also holds the ViT, so it counts as half a GPU."""
        device_map = {}
        world_size = torch.cuda.device_count()
        if "InternVL3" in model_name:
            model_config = AutoConfig.from_pretrained(
                path, revision=revision, trust_remote_code=True
            )
            num_layers = model_config.llm_config.num_hidden_layers
        else:
            num_layers = {
                "InternVL2_5-1B": 24,
                "InternVL2_5-2B": 24,
                "InternVL2_5-4B": 36,
                "InternVL2_5-8B": 32,
                "InternVL2_5-26B": 48,
                "InternVL2_5-38B": 64,
                "InternVL2_5-78B": 80,
            }[model_name]
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        for key in [
            "vision_model",
            "mlp1",
            "language_model.model.tok_embeddings",
            "language_model.model.embed_tokens",
            "language_model.output",
            "language_model.model.norm",
            "language_model.model.rotary_emb",
            "language_model.lm_head",
            f"language_model.model.layers.{num_layers - 1}",
        ]:
            device_map[key] = 0
        return device_map

    def run(
        self, images: list | None = None, texts: list | None = None, prompt: str = ""
    ) -> list[str]:
        if not images and not texts:
            raise ValueError("At least one of images or texts must be provided.")
        batch_size = max(len(images or []), len(texts or []))
        images = images or [None] * batch_size
        texts = texts or [None] * batch_size

        image_inputs, num_patches_list, text_inputs = [], [], []
        for image, text in zip(images, texts, strict=True):
            patches = load_images(image) if image is not None else None
            if patches is not None:
                num_patches_list.append(patches.size(0))
                image_inputs.append(patches)
            else:
                num_patches_list.append(0)
            text_inputs.append(prepare_text_input(text) if text else None)

        pixel_values = None
        if any(img is not None for img in images):
            pixel_values = torch.cat(image_inputs, dim=0).to(torch.bfloat16).cuda()
        else:
            num_patches_list = [0] * batch_size

        questions = [
            prepare_prompt(prompt, text, image is not None)
            for text, image in zip(text_inputs, images, strict=True)
        ]
        try:
            return self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=dict(max_new_tokens=self.max_length, do_sample=False),
            )
        except torch.cuda.OutOfMemoryError:
            return self.handle_oom(batch_size)
        finally:
            self.empty_memory()

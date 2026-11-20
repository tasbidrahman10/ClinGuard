"""Loading the base model and its processor (plan section 6.1; decisions D4, D5).

Primary model: Qwen/Qwen2.5-VL-3B-Instruct at a pinned revision, 4-bit NF4 with double quantization and
fp16 compute on a Kaggle T4 (no bf16 on T4). The image processor is fixed to a constant pixel budget
(min = max = image_pixels; 448 x 448 gives about 240 visual tokens for an IU frontal image).
For CPU tests the same code loads a tiny model of the same architecture without quantization.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"      # A3; pinned at G3
TINY_MODEL_ID = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
TINY_MODEL_REVISION = "a2ae4f40e2b25c3b89a204dc3ed798b4ebc20318"
IMAGE_PIXELS = 448 * 448


def load_processor(model_id=MODEL_ID, revision=MODEL_REVISION, image_pixels=IMAGE_PIXELS):
    """Processor with a fixed pixel budget (min = max = image_pixels).

    The budget is set on the image processor itself and then checked on a synthetic square image. In
    transformers 5.23 a `size=` argument to from_pretrained is silently ignored for this processor (it
    falls back to Qwen's default budget, about 25x more image tokens), and `min_pixels` / `max_pixels`
    are deprecated, so neither is used. The check fails loudly if a library change stops the budget
    from applying."""
    from PIL import Image
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    ip = processor.image_processor
    ip.size = {"shortest_edge": image_pixels, "longest_edge": image_pixels}
    grid = ip(images=[Image.new("RGB", (2048, 2048))], return_tensors="pt")["image_grid_thw"][0].tolist()
    area = grid[1] * grid[2] * ip.patch_size ** 2
    if not 0.7 * image_pixels <= area <= image_pixels:
        raise RuntimeError(f"image budget not applied: {image_pixels} pixels requested, the processor uses {area}")
    return processor


def load_model(model_id=MODEL_ID, revision=MODEL_REVISION, quant="nf4", compute_dtype=torch.float16,
               gradient_checkpointing=True, kbit_prepare=False):
    """quant: "nf4" (GPU, bitsandbytes) or "none" (full precision in compute_dtype, e.g. CPU tests).
    kbit_prepare: run peft's prepare_model_for_kbit_training (upcasts non-quantized weights to fp32;
    more memory, sometimes more stable). Decided by profiling at G3."""
    kwargs = dict(revision=revision)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype)
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = compute_dtype
    elif quant == "none":
        kwargs["dtype"] = compute_dtype
    else:
        raise ValueError(f"unknown quantization '{quant}'")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    if quant == "nf4" and kbit_prepare:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model

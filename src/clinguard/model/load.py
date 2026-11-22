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
# With quantization, modules kept in compute precision when vision_fp16=True. bitsandbytes' default skips only
# the output layer, so without this the whole vision tower and merger are quantized to 4 bit as well.
# Passing a skip list replaces the default, so "lm_head" is listed explicitly.
VISION_SKIP = ("model.visual", "lm_head")


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
               gradient_checkpointing=True, kbit_prepare=False, vision_fp16=False, device_map=None):
    """quant: "nf4" (GPU, bitsandbytes) or "none" (full precision in compute_dtype, e.g. CPU tests).
    vision_fp16: with nf4, keep the vision tower and merger in compute precision (VISION_SKIP; decision I29).
    kbit_prepare: run peft's prepare_model_for_kbit_training (upcasts non-quantized weights to fp32;
    more memory, sometimes more stable). Decided by profiling at G3.
    device_map: e.g. {"": 0} to put an unquantized model on the GPU (nf4 always uses {"": 0})."""
    kwargs = dict(revision=revision)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        extra = dict(llm_int8_skip_modules=list(VISION_SKIP)) if vision_fp16 else {}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype, **extra)
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = compute_dtype
    elif quant == "none":
        kwargs["dtype"] = compute_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
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

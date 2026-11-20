"""Shared fixtures for the model tests (tier b: a tiny model with the real Qwen2.5-VL architecture, on CPU).

Model tests skip when transformers / peft / torchvision are not installed (CI installs only the CPU
engine stack). Data tests skip when the IU X-ray files are not on this machine (they are never committed).
"""
import os

import pytest

RAW_DIR = os.path.join("data", "raw", "iu_xray")
IMAGES_DIR = os.path.join(RAW_DIR, "images", "images_normalized")
TEST_PIXELS = 112 * 112           # 16 visual tokens: fast on CPU; the real runs use 448 x 448


@pytest.fixture(scope="session")
def tiny():
    """(load_fresh_model, processor) for the tiny Qwen2.5-VL test model."""
    for mod in ("transformers", "peft", "torchvision"):
        pytest.importorskip(mod)
    import torch
    from clinguard.model import load
    try:
        processor = load.load_processor(load.TINY_MODEL_ID, load.TINY_MODEL_REVISION, image_pixels=TEST_PIXELS)
    except Exception as e:                                   # offline and not cached
        pytest.skip(f"tiny model not available: {e}")

    def fresh(gradient_checkpointing=False):
        return load.load_model(load.TINY_MODEL_ID, load.TINY_MODEL_REVISION, quant="none",
                               compute_dtype=torch.float32, gradient_checkpointing=gradient_checkpointing)
    return fresh, processor


@pytest.fixture(scope="session")
def iu_data():
    if not os.path.isdir(IMAGES_DIR):
        pytest.skip("IU X-ray files are not on this machine")
    return RAW_DIR, IMAGES_DIR

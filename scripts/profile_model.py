"""Profile the real model on a GPU (plan Phase 3, stage S0): memory and time per optimizer step at each
image budget and micro-batch size, and time per generated report. Also creates the shared LoRA A file
on first use (D5) and a few zero-shot samples for a sanity check.

Usage (Kaggle, one T4):
  CUDA_VISIBLE_DEVICES=0 python scripts/profile_model.py --raw-dir data/raw/iu_xray \
      --images-dir data/raw/iu_xray/images/images_normalized --out /kaggle/working/profile

Writes into --out: profile.json (numbers and environment), lora_A.safetensors (when created), and
zero_shot_samples.json. The samples contain generated and reference report text: keep them private
(Kaggle output or the private HF repo); never commit them.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.data import dataset  # noqa: E402
from clinguard.model import load, lora, params  # noqa: E402
from clinguard.model.generate import generate_findings  # noqa: E402
from clinguard.model.site import VLMSite  # noqa: E402


def versions():
    out = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               platform=platform.platform())
    for mod in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:                                  # noqa: BLE001
            out[mod] = f"unavailable: {e}"
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    try:
        out["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                           # noqa: BLE001
        out["code_commit"] = None
    return out


def time_training(model, processor, examples, micro_batch, grad_accum, steps):
    site = VLMSite("profile", model, processor, examples, micro_batch=micro_batch, grad_accum=grad_accum,
                   lr=1e-4, warmup_steps=0, amp_dtype=torch.float16, seed=0)
    phi0 = site.get_params()
    try:
        site.train(1)                                            # warm-up step (kernels, allocator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        stats = site.train(steps)
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / steps
    finally:
        site.set_params(phi0)                                    # leave the model untrained (zero-shot)
        model.zero_grad(set_to_none=True)
    return dict(micro_batch=micro_batch, grad_accum=grad_accum, effective_batch=micro_batch * grad_accum,
                sec_per_optimizer_step=round(sec, 3), peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                mean_loss=round(stats["loss"], 4), loss_finite=bool(torch.isfinite(torch.tensor(stats["loss"]))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--manifest-dir", default="manifests")
    p.add_argument("--out", required=True)
    p.add_argument("--pixels", type=int, nargs="+", default=[448, 336], help="square side of the pixel budget")
    p.add_argument("--micro", type=int, nargs="+", default=[1, 2, 4], help="micro-batch sizes (effective batch 4)")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--gen", type=int, default=6, help="development examples for the generation timing")
    p.add_argument("--lora-a", default=None, help="existing lora_A.safetensors to verify and load")
    p.add_argument("--lora-a-sha256", default=None)
    p.add_argument("--kbit-prepare", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("profiling needs a GPU")
    os.makedirs(a.out, exist_ok=True)

    dataset.verify_source(a.raw_dir, a.manifest_dir, "skew_v1")
    train = dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir, "skew_v1", "A", "train")[:64]
    dev = dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir, "skew_v1", "A", "development")[:a.gen]

    t0 = time.perf_counter()
    model = load.load_model(quant="nf4", compute_dtype=torch.float16, kbit_prepare=a.kbit_prepare)
    model = lora.apply_fixed_basis_lora(model)
    load_sec = time.perf_counter() - t0
    if a.lora_a:
        lora.load_lora_a(model, a.lora_a, a.lora_a_sha256)
        a_file, a_sha = a.lora_a, a.lora_a_sha256
    else:
        a_file = os.path.join(a.out, "lora_A.safetensors")
        a_sha = lora.save_lora_a(model, a_file)
    ps = params.phi_params(model)
    report = dict(env=versions(), model=load.MODEL_ID, revision=load.MODEL_REVISION, load_sec=round(load_sec, 1),
                  kbit_prepare=a.kbit_prepare, n_phi=sum(x.numel() for x in ps), n_phi_tensors=len(ps),
                  lora_a_file=os.path.basename(a_file), lora_a_sha256=a_sha,
                  memory_after_load_gb=round(torch.cuda.memory_allocated() / 2**30, 2), training=[], generation=[])
    print(json.dumps({k: v for k, v in report.items() if k not in ("training", "generation")}, indent=1), flush=True)

    samples = []
    for side in a.pixels:
        processor = load.load_processor(image_pixels=side * side)
        enc = dataset.build_batch(processor, train[:1])
        tokens = dict(pixels=side * side, sequence_tokens=int(enc["input_ids"].shape[1]),
                      grid_thw=enc["image_grid_thw"].tolist())
        for mb in a.micro:
            try:
                r = time_training(model, processor, train, mb, max(1, 4 // mb), a.steps)
            except torch.cuda.OutOfMemoryError:
                r = dict(micro_batch=mb, error="out of memory")
                torch.cuda.empty_cache()
            r.update(tokens)
            report["training"].append(r)
            print(json.dumps(r), flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts = generate_findings(model, processor, dev, batch_size=2)
        torch.cuda.synchronize()
        g = dict(pixels=side * side, reports=len(dev), sec_per_report=round((time.perf_counter() - t0) / len(dev), 2),
                 mean_words=round(sum(len(t.split()) for t in texts) / len(texts), 1))
        report["generation"].append(g)
        print(json.dumps(g), flush=True)
        samples += [dict(pixels=side * side, uid=e.uid, reference=e.findings, zero_shot=t) for e, t in zip(dev, texts)]

    with open(os.path.join(a.out, "profile.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "zero_shot_samples.json"), "w") as f:
        json.dump(samples, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

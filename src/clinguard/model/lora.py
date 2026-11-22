"""Fixed-basis LoRA (decision D5, plan section 6.2).

LoRA (rank 16, alpha 32, dropout 0) goes on q_proj, k_proj, v_proj and o_proj of the LANGUAGE model only;
no vision-tower module is ever matched. Every A matrix is frozen and identical at all sites and in every
run; only the B matrices train, so for any weights w,  sum_i w_i B_i A = (sum_i w_i B_i) A  and the
weighted average of site updates is exact (manual section 5.3, test A.4).

A is created once from `lora_seed`, saved to a safetensors file with a SHA-256 checksum, and every run
loads that file and verifies the checksum, so A does not depend on library versions or the GPU.

The target set is a run setting (`lora.targets` in the config; default q/k/v/o). A wider set (the MLP
projections gate/up/down) uses an A file made by `extend_lora_a`: the q/k/v/o matrices are copied
unchanged from the primary file and the new ones are drawn with peft's own rule (uniform in
+-1/sqrt(in_features)) from a generator seeded per module name (scripts/make_lora_a.py, decision I28).
"""
import hashlib
import math

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from clinguard.utils.hashing import sha256_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
RANK, ALPHA, LORA_SEED = 16, 32, 2026


def target_module_names(model, targets=TARGETS):
    """Full names of the language-model modules whose last name part is in `targets`, sorted.
    Refuses any vision module (the vision tower has its own q/k/v and MLP projections)."""
    unknown = set(targets) - set(TARGETS) - set(MLP_TARGETS)
    if unknown:
        raise ValueError(f"unsupported LoRA targets {sorted(unknown)}")
    names = sorted(n for n, _ in model.named_modules()
                   if n.rsplit(".", 1)[-1] in targets and "language_model" in n)
    if not names:
        raise ValueError(f"no language-model modules match {list(targets)}")
    if any("visual" in n or "vision" in n for n in names):
        raise ValueError("a vision-tower module matched the LoRA targets")
    return names


def apply_fixed_basis_lora(model, rank=RANK, alpha=ALPHA, lora_seed=LORA_SEED, targets=TARGETS):
    """Wraps `model` with LoRA. A is drawn from `lora_seed` (global random state is left untouched),
    B starts at zero, and only B is trainable. Runs then overwrite A from the shared file (load_lora_a)."""
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=target_module_names(model, targets), init_lora_weights=True)
    with torch.random.fork_rng():
        torch.manual_seed(lora_seed)
        peft_model = get_peft_model(model, cfg)
    for name, p in peft_model.named_parameters():
        p.requires_grad = ".lora_B." in name
    return peft_model


def lora_a_state(model):
    """{name: A} for every LoRA A matrix, fp32 on CPU, sorted by name."""
    return {n: p.detach().float().cpu().contiguous()
            for n, p in sorted(model.named_parameters()) if ".lora_A." in n}


def save_lora_a(model, path):
    """Writes the A matrices and returns the file's SHA-256 (record it in the run config)."""
    save_file(lora_a_state(model), path)
    return sha256_file(path)


def lora_a_specs(model, targets=TARGETS, rank=RANK):
    """{peft parameter name of A: shape} for an UNWRAPPED model, without creating any LoRA module.
    Works on a model built on the meta device, so an A file can be made without the weights."""
    modules = dict(model.named_modules())
    return {f"base_model.model.{n}.lora_A.default.weight": (rank, modules[n].in_features)
            for n in target_module_names(model, targets)}


def _name_seed(lora_seed, name):
    return int(hashlib.sha256(f"{lora_seed}:{name}".encode()).hexdigest()[:15], 16)


def extend_lora_a(base, specs, lora_seed=LORA_SEED):
    """A state for `specs` ({name: shape}): entries present in `base` are copied unchanged (their shapes
    must agree); the others are drawn uniform in +-1/sqrt(in_features), peft's default for A, each from a
    CPU generator seeded by (lora_seed, name), so the result does not depend on module order."""
    out = {}
    for name, shape in sorted(specs.items()):
        if name in base:
            if tuple(base[name].shape) != tuple(shape):
                raise ValueError(f"{name}: base shape {tuple(base[name].shape)} differs from {tuple(shape)}")
            out[name] = base[name].float().contiguous()
            continue
        g = torch.Generator().manual_seed(_name_seed(lora_seed, name))
        bound = 1.0 / math.sqrt(shape[1])
        out[name] = ((torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * bound).float().contiguous()
    extra = set(base) - set(specs)
    if extra:
        raise ValueError(f"the base file has {len(extra)} matrices the new target set does not include")
    return out


def load_lora_a(model, path, expected_sha256):
    """Loads the shared A matrices after checking the file against its recorded checksum."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: checksum {actual[:12]}... does not match the recorded {expected_sha256[:12]}...")
    saved = load_file(path)
    params = {n: p for n, p in model.named_parameters() if ".lora_A." in n}
    if set(saved) != set(params):
        raise ValueError("the A file does not match this model's LoRA modules")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n].to(device=p.device, dtype=p.dtype))

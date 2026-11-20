"""Fixed-basis LoRA (decision D5, plan section 6.2).

LoRA (rank 16, alpha 32, dropout 0) goes on q_proj, k_proj, v_proj and o_proj of the LANGUAGE model only;
no vision-tower module is ever matched. Every A matrix is frozen and identical at all sites and in every
run; only the B matrices train, so for any weights w,  sum_i w_i B_i A = (sum_i w_i B_i) A  and the
weighted average of site updates is exact (manual section 5.3, test A.4).

A is created once from `lora_seed`, saved to a safetensors file with a SHA-256 checksum, and every run
loads that file and verifies the checksum, so A does not depend on library versions or the GPU.
"""
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from clinguard.utils.hashing import sha256_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK, ALPHA, LORA_SEED = 16, 32, 2026


def target_module_names(model):
    """Full names of the language-model attention projections, sorted. Refuses any vision module."""
    names = sorted(n for n, _ in model.named_modules()
                   if n.rsplit(".", 1)[-1] in TARGETS and "language_model" in n)
    if not names:
        raise ValueError("no language-model attention projections found")
    if any("visual" in n or "vision" in n for n in names):
        raise ValueError("a vision-tower module matched the LoRA targets")
    return names


def apply_fixed_basis_lora(model, rank=RANK, alpha=ALPHA, lora_seed=LORA_SEED):
    """Wraps `model` with LoRA. A is drawn from `lora_seed` (global random state is left untouched),
    B starts at zero, and only B is trainable."""
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=target_module_names(model), init_lora_weights=True)
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

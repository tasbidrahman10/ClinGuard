"""The shared trainable vector phi = every LoRA B matrix, flattened in sorted parameter-name order
(plan section 6.2). The engine and controller only ever see this flat fp32 CPU vector.
"""
import torch


def phi_names(model):
    names = sorted(n for n, _ in model.named_parameters() if ".lora_B." in n)
    if not names:
        raise ValueError("model has no LoRA B matrices")
    return names


def phi_params(model):
    """The B parameters themselves, in phi order."""
    named = dict(model.named_parameters())
    return [named[n] for n in phi_names(model)]


def describe(model):
    """[(name, shape)] in phi order; logged with every run."""
    named = dict(model.named_parameters())
    return [(n, tuple(named[n].shape)) for n in phi_names(model)]


def get_phi(params):
    """Flat fp32 CPU copy of the B parameters (pass the list from phi_params)."""
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in params])


def set_phi(params, phi):
    """Copies a flat vector into the B parameters, casting to each parameter's device and dtype."""
    total = sum(p.numel() for p in params)
    if phi.numel() != total:
        raise ValueError(f"phi has {phi.numel()} entries, the model has {total}")
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(phi[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
            offset += n


def expected_phi_size(hidden_size, num_layers, num_heads, num_kv_heads, rank):
    """Number of entries in phi for LoRA on q/k/v/o: B has shape (out_features, rank).
    q and o map to hidden_size; k and v map to num_kv_heads * head_dim."""
    head_dim = hidden_size // num_heads
    return num_layers * rank * (2 * hidden_size + 2 * num_kv_heads * head_dim)

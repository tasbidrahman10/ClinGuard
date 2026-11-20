"""A.4 LoRA identity (manual Appendix A.4; plan decision D5).

With a common fixed A, the weighted average of the sites' low-rank updates is exact:
sum_i w_i B_i A = (sum_i w_i B_i) A. With trainable A_i and B_i, averaging the factors separately is NOT
the same as averaging the products. The second half checks the real fixed-basis LoRA on a tiny model
with the Qwen2.5-VL architecture: only language-model attention is adapted, only B trains, A is
reproducible and file-verified, and phi round-trips exactly.
"""
import pytest
import torch

from clinguard.model.params import expected_phi_size


def _factors(seed, k=3, d_out=6, d_in=5, r=2):
    g = torch.Generator().manual_seed(seed)
    B = [torch.randn(d_out, r, generator=g, dtype=torch.float64) for _ in range(k)]
    A = [torch.randn(r, d_in, generator=g, dtype=torch.float64) for _ in range(k)]
    return B, A, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)


def test_common_basis_weighted_average_is_exact():
    B, A, w = _factors(0)
    A0 = A[0]
    lhs = sum(wi * (Bi @ A0) for wi, Bi in zip(w, B))
    rhs = sum(wi * Bi for wi, Bi in zip(w, B)) @ A0
    assert torch.allclose(lhs, rhs, rtol=1e-13, atol=1e-13)


def test_trainable_factors_do_not_average():
    B, A, w = _factors(1)
    average_of_products = sum(wi * (Bi @ Ai) for wi, Bi, Ai in zip(w, B, A))
    product_of_averages = sum(wi * Bi for wi, Bi in zip(w, B)) @ sum(wi * Ai for wi, Ai in zip(w, A))
    assert (average_of_products - product_of_averages).abs().max() > 1e-2


def test_phi_size_of_the_primary_model():
    # Qwen2.5-VL-3B: hidden 2048, 36 layers, 16 heads, 2 KV heads (config at revision 66285546).
    assert expected_phi_size(2048, 36, 16, 2, 16) == 2_654_208


# ---------------------------------------------------------------- real LoRA on the tiny model

def _lora(tiny, seed=2026):
    from clinguard.model import lora
    fresh, _ = tiny
    return lora.apply_fixed_basis_lora(fresh(), lora_seed=seed)


def test_lora_targets_language_model_attention_only(tiny):
    from clinguard.model import lora
    model = _lora(tiny)
    names = [n for n, _ in model.named_parameters() if ".lora_" in n]
    assert names and all("language_model" in n and "visual" not in n for n in names)
    text_cfg = model.config.text_config
    assert len(lora.target_module_names(tiny[0]())) == 4 * text_cfg.num_hidden_layers


def test_only_b_trains_and_phi_starts_at_zero(tiny):
    from clinguard.model import params
    model = _lora(tiny)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_B." in n for n in trainable)
    phi = params.get_phi(params.phi_params(model))
    c = model.config.text_config
    assert phi.dtype == torch.float32 and torch.count_nonzero(phi) == 0
    assert phi.numel() == expected_phi_size(c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
                                            c.num_key_value_heads, 16)


def test_a_is_reproducible_and_verified_from_file(tiny, tmp_path):
    from clinguard.model import lora
    a1, a2, other = (lora.lora_a_state(_lora(tiny, s)) for s in (2026, 2026, 7))
    assert a1.keys() == a2.keys() and all(torch.equal(a1[k], a2[k]) for k in a1)
    assert any(not torch.equal(a1[k], other[k]) for k in a1)

    path = str(tmp_path / "lora_A.safetensors")
    sha = lora.save_lora_a(_lora(tiny, 2026), path)
    target = _lora(tiny, 7)
    lora.load_lora_a(target, path, sha)
    loaded = lora.lora_a_state(target)
    assert all(torch.equal(a1[k], loaded[k]) for k in a1)
    with pytest.raises(ValueError, match="checksum"):
        lora.load_lora_a(target, path, "0" * 64)


def test_lora_application_leaves_global_random_state_alone(tiny):
    before = torch.get_rng_state()
    _lora(tiny)
    assert torch.equal(before, torch.get_rng_state())


def test_phi_round_trip_and_its_effect_on_a_layer(tiny):
    from clinguard.model import params
    model = _lora(tiny)
    ps = params.phi_params(model)
    phi = torch.randn(sum(p.numel() for p in ps), generator=torch.Generator().manual_seed(0))
    params.set_phi(ps, phi)
    assert torch.equal(params.get_phi(ps), phi)

    name = params.phi_names(model)[0].split(".lora_B.")[0]               # the first adapted projection
    layer = dict(model.named_modules())[name]
    x = torch.randn(3, layer.base_layer.in_features, generator=torch.Generator().manual_seed(1))
    a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    expected = layer.base_layer(x) + layer.scaling["default"] * (x @ a.T @ b.T)
    assert layer.scaling["default"] == 2.0                                # alpha / rank = 32 / 16
    assert torch.allclose(layer(x), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        params.set_phi(ps, phi[:-1])

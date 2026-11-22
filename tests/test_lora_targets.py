"""LoRA target sets and A-file extension (decision I28), on the tiny Qwen2.5-VL (tier b, CPU)."""
import math

import pytest
import torch

ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_specs_name_exactly_the_matrices_peft_creates(tiny):
    from clinguard.model import lora
    fresh, _ = tiny
    for targets in (lora.TARGETS, ALL):
        specs = lora.lora_a_specs(fresh(), targets=targets)
        wrapped = lora.apply_fixed_basis_lora(fresh(), targets=targets)
        created = {n: tuple(p.shape) for n, p in wrapped.named_parameters() if ".lora_A." in n}
        assert specs == created
        assert not any("visual" in n for n in specs)                 # the vision MLPs are never wrapped
    assert any(".mlp.down_proj." in n for n in lora.lora_a_specs(fresh(), targets=ALL))


def test_unknown_targets_are_refused(tiny):
    from clinguard.model import lora
    fresh, _ = tiny
    with pytest.raises(ValueError, match="unsupported"):
        lora.target_module_names(fresh(), ("q_proj", "lm_head"))


def test_extend_keeps_the_base_and_draws_new_matrices_like_peft(tiny):
    from clinguard.model import lora
    fresh, _ = tiny
    base = lora.lora_a_state(lora.apply_fixed_basis_lora(fresh()))
    specs = lora.lora_a_specs(fresh(), targets=ALL)
    ext = lora.extend_lora_a(base, specs)
    assert set(ext) == set(specs)
    assert all(torch.equal(ext[n], base[n]) for n in base)          # q/k/v/o unchanged
    again = lora.extend_lora_a(base, specs)
    assert all(torch.equal(ext[n], again[n]) for n in ext)           # deterministic
    new = [n for n in ext if n not in base]
    assert new and all(ext[n].abs().max() <= 1 / math.sqrt(specs[n][1]) for n in new)
    assert len({ext[n].flatten()[:4].tolist().__repr__() for n in new}) == len(new)   # a different draw per module
    with pytest.raises(ValueError, match="does not include"):
        lora.extend_lora_a(ext, lora.lora_a_specs(fresh()))            # the base has matrices the new set lacks


def test_a_wider_target_set_loads_and_trains_only_b(tiny, tmp_path):
    from safetensors.torch import save_file
    from clinguard.model import lora, params
    from clinguard.utils.hashing import sha256_file
    fresh, _ = tiny
    model = lora.apply_fixed_basis_lora(fresh(), targets=ALL)
    path = str(tmp_path / "a.safetensors")
    save_file(lora.extend_lora_a({}, lora.lora_a_specs(fresh(), targets=ALL)), path)
    lora.load_lora_a(model, path, sha256_file(path))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(trainable) == params.phi_names(model) and any(".mlp." in n for n in trainable)

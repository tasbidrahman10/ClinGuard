"""The real training pipeline on CPU: frozen manifests -> examples -> batches -> VLMSite -> DiLoCo engine.

Uses a tiny model with the Qwen2.5-VL architecture and real IU X-ray images, so every code path the
Kaggle runs use is exercised except 4-bit quantization and fp16. Skips without the model stack or data.
"""
import json

import pytest
import torch

from clinguard.engine.runner import RunConfig, run

PARTITION = "skew_v1"


def _examples(iu_data, site, role, n):
    from clinguard.data import dataset
    raw, images = iu_data
    return dataset.load_examples(raw, images, "manifests", PARTITION, site, role)[:n]


def _sites(tiny, iu_data, gradient_checkpointing=True):
    from clinguard.model import lora
    from clinguard.model.site import VLMSite
    fresh, processor = tiny
    model = lora.apply_fixed_basis_lora(fresh(gradient_checkpointing=gradient_checkpointing))
    return [VLMSite(s, model, processor, _examples(iu_data, s, "train", 6), _examples(iu_data, s, "calibration", 3),
                    micro_batch=1, grad_accum=2, lr=1e-3, warmup_steps=2, seed=100 + i)
            for i, s in enumerate("ABC")], model


def test_examples_come_from_frozen_verified_data(iu_data):
    from clinguard.data import dataset, manifest
    raw, images = iu_data
    m = dataset.verify_source(raw, "manifests", PARTITION)
    ex = dataset.load_examples(raw, images, "manifests", PARTITION, "A", "train")
    assert len(ex) == m["counts"]["A_train"]["total"] == 706
    assert all(e.findings and e.image_path.endswith(".png") for e in ex)
    with pytest.raises(PermissionError):
        dataset.load_examples(raw, images, "manifests", PARTITION, "A", "test")
    assert manifest.read_manifest("manifests", PARTITION)["status"] == "frozen"


def test_labels_supervise_only_the_findings(tiny, iu_data):
    from clinguard.data import dataset
    _, processor = tiny
    ex = _examples(iu_data, "B", "train", 3)
    enc = dataset.build_batch(processor, ex)
    assert enc["input_ids"].shape == enc["labels"].shape
    for row, e in enumerate(ex):
        kept = enc["labels"][row][enc["labels"][row] != dataset.IGNORE_INDEX]
        assert processor.tokenizer.decode(kept) == e.findings + "<|im_end|>\n"
    gen = dataset.build_batch(processor, ex, with_targets=False)
    assert "labels" not in gen and (gen["attention_mask"][:, -1] == 1).all()     # left-padded prompts


def test_calibration_loss_changes_nothing(tiny, iu_data):
    sites, _ = _sites(tiny, iu_data, gradient_checkpointing=False)
    s = sites[0]
    phi, sampler, opt = s.get_params(), s.sampler.state(), s.state_dict()["opt"]
    rng = torch.get_rng_state()
    first, second = s.calibration_loss(), s.calibration_loss()
    assert first == second and first > 0
    assert torch.equal(phi, s.get_params()) and s.sampler.state() == sampler
    assert s.state_dict()["opt"] == opt and torch.equal(rng, torch.get_rng_state())


def test_vlm_sites_train_and_resume_exactly(tiny, iu_data, tmp_path):
    """A.7 on the real pipeline: 3 sites, 2 rounds of H = 2 optimizer steps, loss weighting."""
    cfg = RunConfig(H=2, total_steps_per_site=4, weighting="loss", rho=0.5, method="tiny_smoke")

    def go(out, stop=None):
        sites, model = _sites(tiny, iu_data)                     # a fresh model and fresh sites, as in a new session
        from clinguard.model import params
        phi0 = params.get_phi(params.phi_params(model))
        return run(cfg, sites, phi0, str(tmp_path / out), stop_after_round=stop), sites

    (phi_full, m_full), full_sites = go("full")
    go("part", stop=0)
    (phi_res, m_res), res_sites = go("part")

    assert torch.count_nonzero(phi_full) > 0 and torch.isfinite(phi_full).all()
    assert torch.equal(phi_full, phi_res) and torch.equal(m_full, m_res)
    for a, b in zip(full_sites, res_sites):
        assert a.opt_steps == b.opt_steps == 4
        assert a.sampler.state() == b.sampler.state()
        for sa, sb in zip(a.opt.state_dict()["state"].values(), b.opt.state_dict()["state"].values()):
            assert torch.equal(sa["exp_avg"], sb["exp_avg"]) and torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
    rows = [json.loads(l) for l in open(tmp_path / "full" / "rounds.jsonl")]
    assert [r["round"] for r in rows] == [0, 1]
    assert all(set(r["calib_loss"]) == {"A", "B", "C"} for r in rows)
    meta = json.load(open(tmp_path / "full" / "run.json"))
    assert meta["n_params"] == phi_full.numel() and meta["train_counts"] == [6, 6, 6]


def test_generation_produces_text(tiny, iu_data):
    from clinguard.model import lora
    from clinguard.model.generate import generate_findings
    fresh, processor = tiny
    model = lora.apply_fixed_basis_lora(fresh())
    out = generate_findings(model, processor, _examples(iu_data, "C", "development", 2), max_new_tokens=6)
    assert len(out) == 2 and all(isinstance(t, str) for t in out)

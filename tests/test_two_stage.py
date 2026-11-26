"""A second training stage on a frozen first stage (decision I31, Kaggle session 6), on the tiny Qwen2.5-VL
with real IU images: merging a finished run's LoRA reproduces that run's model, phi is then only the new
stage's B matrices, and a two-stage run goes through run_job and vision_check end to end."""
import importlib.util
import json
import os

import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTITION = "skew_v1"


def _examples(iu_data, n, site="A", role="train"):
    from clinguard.data import dataset
    raw, images = iu_data
    return dataset.load_examples(raw, images, "manifests", PARTITION, site, role)[:n]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vision_stage(fresh, tmp_path, scale=0.05):
    """(phi, A file path, its SHA-256) of a made-up finished vision-only stage."""
    from clinguard.model import lora, params
    wrapped = lora.apply_fixed_basis_lora(fresh(), targets=("vision",))
    a_path = str(tmp_path / "a_vision.safetensors")
    sha = lora.save_lora_a(wrapped, a_path)
    n = sum(p.numel() for p in params.phi_params(wrapped))
    return torch.randn(n, generator=torch.Generator().manual_seed(0)) * scale, a_path, sha


def _losses(model, processor, ex):
    from clinguard.model import diagnostics
    model.eval()
    return [s for s, _ in diagnostics.per_example_losses(model, processor, ex)]


def test_merge_reproduces_the_first_stage(tiny, iu_data, tmp_path):
    from clinguard.model import lora, params
    fresh, processor = tiny
    phi, a_path, sha = _vision_stage(fresh, tmp_path)
    ex = _examples(iu_data, 3)
    adapter = lora.apply_fixed_basis_lora(fresh(), targets=("vision",))
    lora.load_lora_a(adapter, a_path, sha)
    params.set_phi(params.phi_params(adapter), phi)
    expected = _losses(adapter, processor, ex)

    merged = lora.merge_frozen_lora(fresh(), phi, a_path, sha, ("vision",))
    got = _losses(merged, processor, ex)
    base = _losses(fresh(), processor, ex)
    assert got == pytest.approx(expected, abs=1e-3)
    assert max(abs(g - b) for g, b in zip(got, base)) > 1e-2          # the merge did change the model
    assert not any(p.requires_grad for p in merged.parameters())
    assert not any(".lora_" in n for n, _ in merged.named_parameters())

    second = lora.apply_fixed_basis_lora(merged)                        # stage 2: q/k/v/o of the language model
    assert all("language_model" in n for n in params.phi_names(second))
    assert all(".lora_B." in n for n, p in second.named_parameters() if p.requires_grad)


def test_merge_refuses_a_wrong_checksum_and_quantized_layers(tiny, tmp_path):
    from clinguard.model import lora
    fresh, _ = tiny
    phi, a_path, sha = _vision_stage(fresh, tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        lora.merge_frozen_lora(fresh(), phi, a_path, "0" * 64, ("vision",))
    model = fresh()
    first = lora.target_module_names(model, ("vision",))[0]
    layer = model.get_submodule(first)
    layer.weight = torch.nn.Parameter(torch.zeros_like(layer.weight, dtype=torch.uint8), requires_grad=False)
    with pytest.raises(ValueError, match="quantized"):
        lora.merge_frozen_lora(model, phi, a_path, sha, ("vision",))
    rj = _load_script("run_job")
    cfg = dict(model=dict(quant="nf4", pixels=112 * 112), lora={}, vision_stage=dict(run_id="x"))
    with pytest.raises(ValueError, match="vision_fp16"):                # refused before anything is loaded
        rj.build_model(cfg, None, str(tmp_path))


def test_two_stage_run_end_to_end(tiny, iu_data, tmp_path, monkeypatch):
    """Stage 1: vision-only LoRA (report loss, pooled sites). Stage 2: language-model LoRA on site A's
    reports with stage 1 merged and frozen, generation, then vision_check --configs on stage 2."""
    from clinguard.eval.metrics import BLANK, CHEXBERT_CLASSES, NEG, POS
    from clinguard.model import load, lora, params
    from clinguard.utils.hashing import sha256_file
    from clinguard.utils.hf_sync import LocalStore
    monkeypatch.chdir(ROOT)
    fresh, _ = tiny
    rj, vc = _load_script("run_job"), _load_script("vision_check")
    store_dir = str(tmp_path / "store")
    store = LocalStore(store_dir)
    va = str(tmp_path / "av.safetensors")
    vsha = lora.save_lora_a(lora.apply_fixed_basis_lora(fresh(), targets=("vision",)), va)
    ta = str(tmp_path / "at.safetensors")
    tsha = lora.save_lora_a(lora.apply_fixed_basis_lora(fresh()), ta)
    store.upload_files({"shared/a_vision.safetensors": va, "shared/a_text.safetensors": ta}, "A files")
    tiny_model = dict(name=load.TINY_MODEL_ID, revision=load.TINY_MODEL_REVISION, pixels=112 * 112,
                      micro_batch=2, grad_accum=1, lr=1e-2, warmup_steps=1, quant="none")
    engine = dict(H=1, total_steps_per_site=2, eta_out=1.0, mu=0.0, weighting="uniform")

    def run(name, cfg):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path, rj.main(["--config", str(path), "--store-dir", store_dir, "--work-dir", str(tmp_path / "w")])

    _, rid1 = run("stage1", dict(method="eyes", partition=PARTITION, sites=["A+B+C"], seed=11, model=tiny_model,
                                 engine=engine, eval=dict(dev_sites=[]), limits=dict(train=6, calib=2),
                                 lora=dict(targets=["vision"], a_file="shared/a_vision.safetensors", a_sha256=vsha)))
    phi1_remote = f"runs/{rid1}/phi_final.safetensors"
    stage = dict(run_id=rid1, phi_sha256=sha256_file(os.path.join(store_dir, phi1_remote)), targets=["vision"],
                 a_file="shared/a_vision.safetensors", a_sha256=vsha)
    cfg2 = dict(method="twostage", partition=PARTITION, sites=["A"], seed=11, model=tiny_model, engine=engine,
                lora=dict(a_file="shared/a_text.safetensors", a_sha256=tsha), vision_stage=stage,
                eval=dict(dev_sites=["A"], gen_batch=2, max_new_tokens=4), limits=dict(train=4, calib=2, dev=2))
    path2, rid2 = run("stage2", cfg2)
    assert json.loads(store.read_bytes(f"runs/{rid2}/LEASE.json"))["status"] == "finished"
    assert len(store.read_bytes(f"runs/{rid2}/generations/dev_A.jsonl").decode().splitlines()) == 2
    run_json = store.read_bytes(f"runs/{rid2}/run.json").decode()
    assert rid1 in run_json and stage["phi_sha256"] in run_json           # the frozen stage is recorded

    # the stage-2 model = stage 1 merged + its own language-model LoRA, which is all of phi
    from safetensors.torch import load_file
    _, model2, _ = rj.build_model(cfg2, store, str(tmp_path / "w3"))
    phi2 = load_file(os.path.join(store_dir, f"runs/{rid2}/phi_final.safetensors"))["phi"]
    assert phi2.numel() == sum(p.numel() for p in params.phi_params(lora.apply_fixed_basis_lora(fresh())))
    params.set_phi(params.phi_params(model2), phi2)
    stage1 = lora.merge_frozen_lora(fresh(), load_file(os.path.join(store_dir, phi1_remote))["phi"], va, vsha, ("vision",))
    vis2 = {n.removeprefix("base_model.model."): p for n, p in model2.named_parameters() if ".visual." in n}
    vis1 = {n: p for n, p in stage1.named_parameters() if ".visual." in n}
    assert set(vis2) == set(vis1) and all(torch.equal(vis2[n], vis1[n]) for n in vis1)
    assert any(not torch.equal(vis1[n], p) for n, p in fresh().named_parameters() if n in vis1)

    with pytest.raises(ValueError, match="checksum"):                   # a different stage-1 file is refused
        rj.build_model({**cfg2, "vision_stage": {**stage, "phi_sha256": "0" * 64}}, store, str(tmp_path / "w4"))

    dev = [e for s in "ABC" for e in _examples(iu_data, 4, s, "development")]
    labels = {}
    for k, e in enumerate(dev):
        v = [BLANK] * len(CHEXBERT_CLASSES)
        v[CHEXBERT_CLASSES.index("Cardiomegaly")] = (POS, NEG)[k % 2]
        labels[e.uid] = v
    store.write_bytes(vc.LABELS_PATH, json.dumps(labels).encode(), "labels")
    report = vc.main(["--name", "t", "--store-dir", store_dir, "--variants", "--configs", str(path2), "--n", "4",
                      "--batch", "4", "--pixels", str(112 * 112), "--model", load.TINY_MODEL_ID,
                      "--revision", load.TINY_MODEL_REVISION, "--work-dir", str(tmp_path / "w5"),
                      "--out-dir", str(tmp_path / "o")])
    assert report["results"][rid2]["probes"]["Cardiomegaly"]["n_present"] == 6

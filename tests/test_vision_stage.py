"""Teaching the vision side (decision I30, Kaggle session 5) on the tiny Qwen2.5-VL with real IU images:
the "vision" LoRA target, the finding-sentence objective, and run_job/vision_check end to end."""
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


def test_vision_target_is_every_vision_linear_layer(tiny):
    from clinguard.model import lora
    fresh, _ = tiny
    model = fresh()
    vision = sorted(n for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and n.startswith("model.visual"))
    assert lora.target_module_names(model, ("vision",)) == vision
    assert any(".merger." in n for n in vision) and any(".blocks." in n for n in vision)
    both = lora.target_module_names(model, ("vision", "q_proj"))
    assert set(vision) < set(both) and all("language_model" in n for n in set(both) - set(vision))
    wrapped = lora.apply_fixed_basis_lora(fresh(), targets=("vision",))
    created = {n: tuple(p.shape) for n, p in wrapped.named_parameters() if ".lora_A." in n}
    assert created == lora.lora_a_specs(fresh(), targets=("vision",))
    assert not any("language_model" in n for n in created)


def test_only_the_vision_side_trains(tiny, iu_data):
    """Report loss, vision-only LoRA, gradient checkpointing on: the vision B matrices move, nothing else."""
    from clinguard.model import lora, params
    from clinguard.model.site import VLMSite
    fresh, processor = tiny
    model = lora.apply_fixed_basis_lora(fresh(gradient_checkpointing=True), targets=("vision",))
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    site = VLMSite("A", model, processor, _examples(iu_data, 4), micro_batch=2, grad_accum=1, lr=1e-2,
                   warmup_steps=1, seed=0)
    phi0 = site.get_params()
    site.train(2)
    assert not torch.equal(site.get_params(), phi0)
    assert all(".visual." in n for n in params.phi_names(model))
    assert all(torch.equal(p, frozen_before[n]) for n, p in model.named_parameters() if n in frozen_before)


def _labels_for(examples, finding_values):
    from clinguard.eval.metrics import BLANK, CHEXBERT_CLASSES
    out = {}
    for k, e in enumerate(examples):
        v = [BLANK] * len(CHEXBERT_CLASSES)
        for finding, values in finding_values.items():
            v[CHEXBERT_CLASSES.index(finding)] = values[k % len(values)]
        out[e.uid] = v
    return out


def test_finding_pairs_are_balanced(iu_data):
    from clinguard.eval.metrics import NEG, POS
    from clinguard.model.finding_site import finding_pairs
    ex = _examples(iu_data, 60)
    # Cardiomegaly: 1 present in 6 -> 10 present, 50 absent; Edema: never present -> skipped
    labels = _labels_for(ex, {"Cardiomegaly": [POS, NEG, NEG, NEG, NEG, NEG], "Edema": [NEG]})
    pairs, table = finding_pairs(ex, labels, min_count=5, max_ratio=4, seed=1)
    card = [p for p in pairs if p.finding == "Cardiomegaly"]
    assert table["Cardiomegaly"]["used"] and not table["Edema"]["used"]
    assert table["Cardiomegaly"]["absent"] == 40                       # at most 4x the 10 present
    present = [p for p in card if p.present]
    assert len(present) == 40 and len(card) - len(present) == 40         # present repeated 4x: balanced
    assert all(p.correct == "The heart is enlarged." for p in present)
    assert all(p.correct == "The heart size is normal." and p.wrong == "The heart is enlarged."
               for p in card if not p.present)
    assert finding_pairs(ex, labels, min_count=5, seed=1)[0] == pairs      # deterministic


def test_pair_gap_matches_the_per_study_losses(tiny, iu_data):
    from clinguard.model import diagnostics, lora
    from clinguard.model.finding_site import FindingPair, pair_logprob_gap
    import contextlib
    import dataclasses
    fresh, processor = tiny
    model = lora.apply_fixed_basis_lora(fresh())
    model.eval()
    ex = _examples(iu_data, 3)
    pairs = [FindingPair(e.uid, e.image_path, "Cardiomegaly", "The heart is enlarged.", "The heart size is normal.", True)
             for e in ex]
    with torch.no_grad():
        gap = pair_logprob_gap(model, processor, pairs, torch.device("cpu"), contextlib.nullcontext())
    right = diagnostics.per_example_losses(model, processor, [dataclasses.replace(e, findings=p.correct) for e, p in zip(ex, pairs)])
    wrong = diagnostics.per_example_losses(model, processor, [dataclasses.replace(e, findings=p.wrong) for e, p in zip(ex, pairs)])
    expected = torch.tensor([w[0] - r[0] for r, w in zip(right, wrong)])
    assert torch.allclose(gap, expected, atol=1e-3)


def test_findings_run_end_to_end_and_its_vision_check(tiny, iu_data, tmp_path, monkeypatch):
    """Pooled site A+B+C, vision-only LoRA, finding-sentence objective, through run_job and the store;
    then vision_check --configs reads the run's phi_final."""
    from clinguard.eval.metrics import NEG, POS
    from clinguard.model import load, lora
    from clinguard.utils.hf_sync import LocalStore
    monkeypatch.chdir(ROOT)
    fresh, _ = tiny

    def load_script(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    rj, vc = load_script("run_job"), load_script("vision_check")

    store_dir = str(tmp_path / "store")
    store = LocalStore(store_dir)
    a_path = str(tmp_path / "a.safetensors")
    sha = lora.save_lora_a(lora.apply_fixed_basis_lora(fresh(), targets=("vision",)), a_path)
    store.upload_files({"shared/a_vision.safetensors": a_path}, "A")
    train = [e for s in "ABC" for e in _examples(iu_data, 12, s)]
    store.write_bytes("labels/train.json", json.dumps(_labels_for(train, {"Cardiomegaly": [POS, NEG, NEG]})).encode(), "l")
    dev = [e for s in "ABC" for e in _examples(iu_data, 4, s, "development")]
    store.write_bytes(vc.LABELS_PATH, json.dumps(_labels_for(dev, {"Cardiomegaly": [POS, NEG]})).encode(), "l")

    cfg = dict(method="eyes", partition=PARTITION, sites=["A+B+C"], seed=11, objective="findings",
               finding_labels="labels/train.json", finding_pairs=dict(min_count=2),
               model=dict(name=load.TINY_MODEL_ID, revision=load.TINY_MODEL_REVISION, pixels=112 * 112,
                          micro_batch=2, grad_accum=1, lr=1e-3, warmup_steps=1, quant="none"),
               engine=dict(H=1, total_steps_per_site=2, eta_out=1.0, mu=0.0, weighting="uniform"),
               lora=dict(targets=["vision"], a_file="shared/a_vision.safetensors", a_sha256=sha),
               eval=dict(dev_sites=[]), limits=dict(train=12, calib=2))
    path = tmp_path / "eyes.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rid = rj.main(["--config", str(path), "--store-dir", store_dir, "--work-dir", str(tmp_path / "w")])
    assert json.loads(store.read_bytes(f"runs/{rid}/LEASE.json"))["status"] == "finished"
    assert len(store.read_bytes(f"runs/{rid}/rounds.jsonl").decode().splitlines()) == 2

    report = vc.main(["--name", "t", "--store-dir", store_dir, "--variants", "--configs", str(path), "--n", "4",
                      "--batch", "4", "--pixels", str(112 * 112), "--model", load.TINY_MODEL_ID,
                      "--revision", load.TINY_MODEL_REVISION, "--work-dir", str(tmp_path / "w2"),
                      "--out-dir", str(tmp_path / "o")])
    card = report["results"][rid]["probes"]["Cardiomegaly"]
    assert card["n_present"] == 6 and card["n_absent"] == 6 and 0.0 <= card["auroc"] <= 1.0

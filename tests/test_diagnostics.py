"""Image test (gate G3): diagnostics module and scripts/image_test.py, on the tiny Qwen2.5-VL with real IU
images (tier b, CPU)."""
import importlib.util
import json
import os

import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTITION = "skew_v1"


def _examples(iu_data, n, site="A", role="development"):
    from clinguard.data import dataset
    raw, images = iu_data
    return dataset.load_examples(raw, images, "manifests", PARTITION, site, role)[:n]


def test_image_conditions(iu_data, tmp_path):
    from clinguard.model import diagnostics as D
    ex = _examples(iu_data, 5)
    shuffled = D.with_images(ex, "shuffled")
    assert [e.findings for e in shuffled] == [e.findings for e in ex]           # same reports
    assert all(s.image_path != e.image_path for s, e in zip(shuffled, ex))      # every image replaced
    assert sorted(s.image_path for s in shuffled) == sorted(e.image_path for e in ex)
    blank = str(tmp_path / "blank.png")
    D.write_blank_image(blank)
    assert {e.image_path for e in D.with_images(ex, "blank", blank)} == {blank}
    with pytest.raises(ValueError):
        D.with_images(ex[:1], "shuffled")


def test_per_example_losses_add_up_to_the_model_loss(tiny, iu_data):
    """Token-weighted mean of the per-study losses = the loss the training code computes (same batches)."""
    from clinguard.model import diagnostics as D
    from clinguard.model import lora
    from clinguard.model.site import VLMSite
    fresh, processor = tiny
    model = lora.apply_fixed_basis_lora(fresh())
    ex = _examples(iu_data, 6)
    rows = D.per_example_losses(model, processor, ex, batch_size=4)
    assert len(rows) == 6 and all(t > 0 and l > 0 for l, t in rows)
    site = VLMSite("A", model, processor, ex, ex, micro_batch=1, grad_accum=1, calib_batch=4)
    assert sum(l for l, _ in rows) / sum(t for _, t in rows) == pytest.approx(site.calibration_loss(), rel=1e-5)
    summ = D.summarize(ex, {"true": rows, "shuffled": rows})
    assert summ["all"]["n"] == 6 and summ["all"]["share_true_image_better"] == 0.0
    assert summ["all"]["mean_gain_true_vs_shuffled"] == 0.0


def test_auroc():
    from clinguard.model.diagnostics import auroc
    assert auroc([3, 4], [1, 2]) == 1.0 and auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 3], [2, 2]) == 0.5 and auroc([2], [2]) == 0.5 and auroc([], [1]) is None


def test_vision_skip_list_keeps_exactly_the_vision_tower_unquantized(tiny):
    """Static check of decision I29 on the real architecture's module names (tiny model, CPU)."""
    from transformers.quantizers.base import get_keys_to_not_convert
    from transformers.quantizers.quantizers_utils import should_convert_module
    from clinguard.model import load
    fresh, _ = tiny
    model = fresh()
    linear = [n for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)]
    vision = [n for n in linear if n.startswith("model.visual")]
    assert vision and all(should_convert_module(n, get_keys_to_not_convert(model)) for n in vision)  # the old default
    kept = [n for n in linear if not should_convert_module(n, list(load.VISION_SKIP))]
    assert sorted(kept) == sorted(vision + ["lm_head"])


def test_vision_check_script(tiny, iu_data, tmp_path, monkeypatch):
    from clinguard.eval.metrics import CHEXBERT_CLASSES, POS, NEG, BLANK
    from clinguard.model import load
    from clinguard.utils.hf_sync import LocalStore
    monkeypatch.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("vision_check", os.path.join(ROOT, "scripts", "vision_check.py"))
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    store_dir = str(tmp_path / "store")
    labels = {}
    for site in ("A", "B", "C"):
        for k, e in enumerate(_examples(iu_data, 4, site)):
            v = [BLANK] * len(CHEXBERT_CLASSES)
            v[CHEXBERT_CLASSES.index("Cardiomegaly")] = POS if k % 2 else NEG
            labels[e.uid] = v
    LocalStore(store_dir).write_bytes(vc.LABELS_PATH, json.dumps(labels).encode(), "labels")
    report = vc.main(["--name", "t", "--store-dir", store_dir, "--variants", "cpu_fp32", "--n", "4", "--batch", "4",
                      "--pixels", str(112 * 112), "--model", load.TINY_MODEL_ID, "--revision", load.TINY_MODEL_REVISION,
                      "--work-dir", str(tmp_path / "w"), "--out-dir", str(tmp_path / "o")])
    card = report["results"]["cpu_fp32"]["probes"]["Cardiomegaly"]
    assert card["n_present"] == 6 and card["n_absent"] == 6 and 0.0 <= card["auroc"] <= 1.0
    assert report["results"]["cpu_fp32"]["probes"]["Pleural Effusion"]["auroc"] is None      # no labels: no AUROC
    assert report["results"]["cpu_fp32"]["image_test"]["all"]["n"] == 4
    assert json.loads(LocalStore(store_dir).read_bytes("diagnostics/vision_check_t.json"))["n_dev"] == 12


def test_image_test_script(tiny, iu_data, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from clinguard.model import load, lora, params
    from clinguard.utils.hf_sync import LocalStore
    monkeypatch.chdir(ROOT)
    fresh, _ = tiny
    spec = importlib.util.spec_from_file_location("image_test", os.path.join(ROOT, "scripts", "image_test.py"))
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)
    rj = it.run_job

    store_dir = str(tmp_path / "store")
    store = LocalStore(store_dir)
    a_path = str(tmp_path / "a.safetensors")
    model = lora.apply_fixed_basis_lora(fresh())
    sha = lora.save_lora_a(model, a_path)
    store.upload_files({"shared/a.safetensors": a_path}, "A")
    base = dict(partition=PARTITION, sites=["A"], seed=11,
                model=dict(name=load.TINY_MODEL_ID, revision=load.TINY_MODEL_REVISION, pixels=112 * 112,
                           micro_batch=1, grad_accum=1, lr=1e-3, warmup_steps=1, quant="none"),
                engine=dict(H=1, total_steps_per_site=1, eta_out=1.0, mu=0.0, weighting="uniform"),
                lora=dict(a_file="shared/a.safetensors", a_sha256=sha), eval=dict(dev_sites=[]))
    trained = dict(base, method="trained")
    missing = dict(base, method="not_run_yet")
    paths = []
    for cfg in (trained, missing):
        path = tmp_path / f"{cfg['method']}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        paths.append(str(path))
    phi = 0.05 * torch.randn(sum(p.numel() for p in params.phi_params(model)), generator=torch.Generator().manual_seed(0))
    save_file({"phi": phi}, str(tmp_path / "phi.safetensors"))
    store.upload_files({f"runs/{rj.run_id_for(trained)}/phi_final.safetensors": str(tmp_path / "phi.safetensors")}, "phi")

    report = it.main(["--configs", *paths, "--name", "t", "--store-dir", store_dir, "--n", "6",
                      "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")])
    assert set(report["results"]) == {"base", rj.run_id_for(trained)}
    assert report["skipped"] == [rj.run_id_for(missing)]
    for r in report["results"].values():
        assert all(r["all"][f"loss_{c}"] > 0 for c in ("true", "shuffled", "blank"))
    assert report["results"]["base"]["all"]["loss_true"] != report["results"][rj.run_id_for(trained)]["all"]["loss_true"]
    assert json.loads(store.read_bytes("diagnostics/image_test_t.json"))["n"] == 6

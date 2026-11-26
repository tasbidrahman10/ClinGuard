"""Diagnosis prompt (decision I32, Kaggle session 7): the hint sentence in the prompt, the finding classifier
(image features, logistic regression, cut-offs), the constant-report baseline and the label-level check, and a
classifier + prompted run end to end on the tiny Qwen2.5-VL with real IU images."""
import importlib.util
import json
import os

import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTITION = "skew_v1"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _labels(**marks):
    from clinguard.eval.metrics import BLANK, CHEXBERT_CLASSES
    v = [BLANK] * len(CHEXBERT_CLASSES)
    for name, value in marks.items():
        v[CHEXBERT_CLASSES.index(name.replace("_", " "))] = value
    return v


# ------------------------------------------------------------------------------------------- prompt

def test_hint_sentence_and_prompt_v1_unchanged():
    from clinguard.data import dataset
    assert dataset.messages()[1]["content"][1]["text"] == dataset.USER_PROMPT          # no hint = prompt_v1
    assert dataset.hint_text(("Support Devices", "Cardiomegaly")) == \
        "Findings suggested by an image classifier: cardiomegaly, medical device."
    assert dataset.hint_text(()) == "Findings suggested by an image classifier: none."
    assert dataset.messages(hint=())[1]["content"][1]["text"].startswith(dataset.USER_PROMPT + " Findings")
    with pytest.raises(ValueError, match="hint words"):
        dataset.hint_text(("Pneumothorax",))


def test_hinted_batches_supervise_only_the_answer(tiny, iu_data):
    from dataclasses import replace
    from clinguard.data import dataset
    _, processor = tiny
    raw, images = iu_data
    ex = dataset.load_examples(raw, images, "manifests", PARTITION, "A", "train")[:2]
    ex = [replace(ex[0], hint=("Cardiomegaly",)), replace(ex[1], hint=())]
    enc = dataset.build_batch(processor, ex)
    tok = processor.tokenizer
    for row, e in enumerate(ex):
        ids = enc["input_ids"][row]
        text = tok.decode(ids)
        assert dataset.hint_text(e.hint) in text
        supervised = tok.decode(ids[enc["labels"][row] != dataset.IGNORE_INDEX])
        assert supervised.startswith(e.findings) and dataset.HINT_PREFIX not in supervised
    gen = dataset.build_batch(processor, ex, with_targets=False)
    texts = [tok.decode(r) for r in gen["input_ids"]]
    assert "cardiomegaly." in texts[0] and "classifier: none." in texts[1]


# --------------------------------------------------------------------------------------- classifier

def test_cutoff_counts_every_unstated_finding_as_wrong():
    from clinguard.model.classifier import choose_cutoff
    assert choose_cutoff([3, 2, 1, 0], [1, 1, 0, 0]) == dict(cutoff=2.0, net=2, correct=2, wrong=0)
    assert choose_cutoff([3, 2, 1], [0, 0, 1])["cutoff"] is None                  # never more right than wrong
    assert choose_cutoff([2, 2, 1], [1, 0, 1]) == dict(cutoff=1.0, net=1, correct=2, wrong=1)   # no cut inside a tie


def test_classifier_learns_a_signal_and_skips_uncertain_labels():
    from clinguard.eval.metrics import NEG, POS, UNC
    from clinguard.model import classifier
    g = torch.Generator().manual_seed(0)

    def split(n, offset):
        x = torch.randn(n, 6, generator=g)
        uids = [str(offset + i) for i in range(n)]
        labels = {}
        for i, u in enumerate(uids):
            cm = POS if x[i, 0] > 1.0 else NEG
            labels[u] = _labels(Cardiomegaly=UNC if i % 10 == 0 else cm, Support_Devices=POS if i % 4 == 0 else "")
        return uids, x, labels
    train, calib = split(400, 0), split(200, 1000)
    clf = classifier.fit_classifiers(train, calib, findings=("Cardiomegaly", "Support Devices"))
    cm = clf["models"]["Cardiomegaly"]
    assert cm["train_n"] == 360 and cm["calib_n"] == 180                         # uncertain studies left out
    assert cm["calib_auroc"] > 0.95 and cm["cutoff"] is not None and cm["net"] > 0
    again = classifier.fit_classifiers(train, calib, findings=("Cardiomegaly", "Support Devices"))
    assert torch.equal(again["models"]["Cardiomegaly"]["w"], cm["w"])           # deterministic
    pred = classifier.predict(clf, calib[0], calib[1])
    hits = [u for u, f in pred.items() if "Cardiomegaly" in f]
    assert hits and all(calib[1][calib[0].index(u), 0] > 0.5 for u in hits)
    rows = classifier.summary(clf, calib)
    assert rows["Cardiomegaly"]["dev_correct"] == cm["correct"]


def test_image_features_shape_and_peft_wrapper(tiny, iu_data):
    from clinguard.data import dataset
    from clinguard.model import classifier, lora
    fresh, processor = tiny
    raw, images = iu_data
    ex = dataset.load_examples(raw, images, "manifests", PARTITION, "A", "train")[:3]
    model = fresh()
    feats = classifier.image_features(model, processor, ex, batch_size=2)
    assert feats.shape == (3, 2 * model.config.text_config.hidden_size) and torch.isfinite(feats).all()
    assert not torch.allclose(feats[0], feats[1])                                # different images differ
    wrapped = classifier.image_features(lora.apply_fixed_basis_lora(model), processor, ex, batch_size=3)
    assert torch.allclose(wrapped, feats, atol=1e-5)                             # LoRA B = 0, batch size free


# ------------------------------------------------------------------------------------------ metrics

def test_constant_report_baseline_and_label_level_check():
    from clinguard.eval import metrics
    from clinguard.eval.metrics import NEG, POS
    devices_only = _labels(Support_Devices=POS)
    assert metrics.always_normal_error(devices_only) and not metrics.constant_report_error(devices_only)
    assert metrics.constant_report_error(_labels(Atelectasis=NEG))
    assert not metrics.constant_report_error(_labels(Cardiomegaly=NEG))         # unrelated negatives are free
    refs = {"1": _labels(Cardiomegaly=POS), "2": _labels(Cardiomegaly=NEG), "3": _labels(), "4": devices_only}
    chk = metrics.label_level_check({"1": ["Cardiomegaly"], "2": ["Cardiomegaly"], "3": [], "4": []}, refs)
    assert chk["n"] == 4 and chk["studies_with_a_hint"] == 2
    assert chk["error_rate"] == 0.5 and chk["always_normal_error_rate"] == 0.5    # fixes 1, breaks 2, misses 4
    assert chk["vs_always_normal"] == dict(model_only_errors=1, baseline_only_errors=1, p_one_sided=0.75)
    s = metrics.summarize([dict(site="A", needs_prior=False, reference="x", generated="y", proxy_error=False,
                                ref_labels=devices_only, hyp_labels=devices_only, radgraph=[0, 0, 0], rouge_l=0)], set())
    assert s["constant_report"] == list(metrics.CONSTANT_REPORT_V1)
    assert s["subsets"]["all"]["vs_constant_report"]["baseline_only_errors"] == 0


# ------------------------------------------------------------------------------------------ end to end

def test_classifier_and_prompted_run_end_to_end(tiny, iu_data, tmp_path, monkeypatch):
    from clinguard.data import dataset
    from clinguard.eval.metrics import NEG, POS, UNC
    from clinguard.model import load, lora
    from clinguard.model.classifier import true_hint
    from clinguard.utils.hf_sync import LocalStore
    monkeypatch.chdir(ROOT)
    fresh, _ = tiny
    raw, images = iu_data
    rj, cl = _load_script("run_job"), _load_script("classify")
    store_dir = str(tmp_path / "store")
    store = LocalStore(store_dir)
    ta = str(tmp_path / "at.safetensors")
    tsha = lora.save_lora_a(lora.apply_fixed_basis_lora(fresh()), ta)
    store.upload_files({"shared/a_text.safetensors": ta}, "A file")
    paths = {"train": "labels/chexbert_train.json", "calibration": "labels/chexbert_calibration.json",
             "development": "diagnostics/dev_ref_labels_chexbert.json"}
    for role, path in paths.items():
        uids = [e.uid for s in "ABC" for e in dataset.load_examples(raw, images, "manifests", PARTITION, s, role)]
        lab = {u: _labels(**{f.replace(" ", "_"): POS if (k + j) % 4 == 0 else NEG
                             for j, f in enumerate(dataset.HINT_WORDS)}) for k, u in enumerate(uids)}
        for k, u in enumerate(uids):
            lab[u][1] = (POS, NEG, UNC)[k % 3]                                     # Cardiomegaly, with uncertain
        store.write_bytes(path, json.dumps(lab).encode(), role)
    findings = list(dataset.HINT_WORDS)
    cfg = dict(method="prompted", partition=PARTITION, sites=["A"], seed=11,
               model=dict(name=load.TINY_MODEL_ID, revision=load.TINY_MODEL_REVISION, pixels=112 * 112,
                          micro_batch=2, grad_accum=1, lr=1e-2, warmup_steps=1, quant="none"),
               engine=dict(H=1, total_steps_per_site=2, eta_out=1.0, mu=0.0, weighting="uniform"),
               lora=dict(a_file="shared/a_text.safetensors", a_sha256=tsha),
               diagnosis_prompt=dict(findings=findings, train_labels=paths["train"], calib_labels=paths["calibration"],
                                     predictions="classifier/t/predictions.json", oracle_labels=paths["development"],
                                     wait_minutes=0),
               eval=dict(dev_sites=["A"], gen_batch=2, max_new_tokens=4), limits=dict(train=4, calib=2, dev=2))
    cfg_path = tmp_path / "prompted.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run = ["--config", str(cfg_path), "--store-dir", store_dir, "--work-dir", str(tmp_path / "w")]

    with pytest.raises(SystemExit, match="still not in the store"):               # writer waits for the classifier
        rj.main(run)
    assert json.loads(store.read_bytes(f"runs/{rj.run_id_for(cfg)}/LEASE.json"))["status"] == "paused"
    report = cl.main(["--config", str(cfg_path), "--name", "t", "--store-dir", store_dir, "--limit", "6",
                      "--batch", "4", "--work-dir", str(tmp_path / "wc")])
    assert set(report["findings"]) == set(findings) and report["n"] == dict(train=18, calibration=18, development=18)
    assert "label_level_check" in report
    pred = json.loads(store.read_bytes("classifier/t/predictions.json"))
    assert pred["findings"] == findings and len(pred["development"]) == 18
    again = cl.main(["--config", str(cfg_path), "--name", "t", "--store-dir", store_dir, "--limit", "6",
                     "--work-dir", str(tmp_path / "wc")])
    assert again == report                                                        # not recomputed

    rid = rj.main(run)                                       # resumes: training done, now the reports
    assert json.loads(store.read_bytes(f"runs/{rid}/LEASE.json"))["status"] == "finished"
    dev_labels = json.loads(store.read_bytes(paths["development"]))
    for folder in ("generations", "generations_oracle"):
        rows = [json.loads(line) for line in store.read_bytes(f"runs/{rid}/{folder}/dev_A.jsonl").decode().splitlines()]
        assert len(rows) == 2
        for r in rows:
            expected = pred["development"][r["uid"]] if folder == "generations" else list(true_hint(dev_labels[r["uid"]], findings))
            assert r["hint"] == expected and len(r["hint_source"]["sha256"]) == 64

"""Finding classifier for the diagnosis prompt (decision I32, Kaggle session 7).

  python scripts/classify.py --hf-repo tasbid66/clinguard-runs --config configs/experiment/g3_A_prompted.yaml --name eyes_v1

Builds the model of a run config (base model + its frozen vision stage; the run's own LoRA plays no part),
pools image features for the training, calibration and development studies of every site, fits one logistic
regression per finding on the training studies and picks the cut-offs on the calibration studies
(clinguard.model.classifier). Writes to classifier/<name>/ in the PRIVATE store:
  features.safetensors     pooled image features per role (uids in the metadata)
  classifier.safetensors   standardization and weights per finding
  predictions.json         {uid: [findings]} for the calibration and development studies, plus the cut-offs
  report.json              numbers only: per finding AUROC and flags; the label-level check (below)
Label-level check: proxy_v1 result on the development studies if each report asserted exactly its predicted
findings (a writer that copies its prompt perfectly), against always-normal and the constant report, and
with the predictions shuffled between studies (what is left once the image is ignored).
If predictions.json is already in the store, nothing is recomputed (--force to redo).
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.utils import hf_sync  # noqa: E402

ROLES = ("train", "calibration", "development")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="run config whose model (and vision stage) gives the features")
    p.add_argument("--name", required=True)
    p.add_argument("--hf-repo")
    p.add_argument("--store-dir")
    p.add_argument("--train-labels", default="labels/chexbert_train.json")
    p.add_argument("--calib-labels", default="labels/chexbert_calibration.json")
    p.add_argument("--dev-labels", default="diagnostics/dev_ref_labels_chexbert.json",
                   help="for the report only (AUROC and the label-level check); not used for fitting")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="first n studies per role and site (tests)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--work-dir", default="runs_work")
    p.add_argument("--raw-dir", default="data/raw/iu_xray")
    p.add_argument("--images-dir", default="data/raw/iu_xray/images/images_normalized")
    p.add_argument("--manifest-dir", default="manifests")
    a = p.parse_args(argv)
    if bool(a.hf_repo) == bool(a.store_dir):
        p.error("give exactly one of --hf-repo and --store-dir")
    store = hf_sync.HFStore(a.hf_repo) if a.hf_repo else hf_sync.LocalStore(a.store_dir)
    prefix = f"classifier/{a.name}"
    if store.exists(f"{prefix}/predictions.json") and not a.force:
        report = json.loads(store.read_bytes(f"{prefix}/report.json"))
        print(f"{prefix} is already in the store; nothing recomputed")
        print_report(report)
        return report

    import torch
    from safetensors.torch import load_file, save_file
    import run_job
    from clinguard.data import dataset
    from clinguard.eval import metrics
    from clinguard.model import classifier

    cfg = run_job.load_config(a.config)
    m = dataset.verify_source(a.raw_dir, a.manifest_dir, cfg["partition"])
    examples = {}
    for role in ROLES:
        examples[role] = [e for s in m["sites"] for e in dataset.load_examples(
            a.raw_dir, a.images_dir, a.manifest_dir, cfg["partition"], s, role)[:a.limit]]
    out_dir = os.path.join(a.work_dir, "classifier", a.name)
    os.makedirs(out_dir, exist_ok=True)

    feat_path = os.path.join(out_dir, "features.safetensors")
    if store.exists(f"{prefix}/features.safetensors") and not a.force:
        store.download(f"{prefix}/features.safetensors", feat_path)
        print("features from the store", flush=True)
    else:
        processor, model, amp = run_job.build_model(cfg, store, a.work_dir)
        feats = {}
        for role in ROLES:
            feats[role] = classifier.image_features(model, processor, examples[role], a.batch, amp)
            print(f"{role}: features of {len(examples[role])} studies, {feats[role].shape[1]} numbers each", flush=True)
        del model
        save_file({r: f.contiguous() for r, f in feats.items()}, feat_path,
                  metadata={r: json.dumps([e.uid for e in examples[r]]) for r in ROLES})
        store.upload_files({f"{prefix}/features.safetensors": feat_path}, f"{prefix}: image features")
    feats = load_file(feat_path)
    from safetensors import safe_open
    with safe_open(feat_path, "pt") as f:
        uids = {r: json.loads(f.metadata()[r]) for r in ROLES}
    if any(uids[r] != [e.uid for e in examples[r]] for r in ROLES):
        raise SystemExit("the stored features belong to other studies (different --limit?); use --force")

    labels = {"train": json.loads(store.read_bytes(a.train_labels)),
              "calibration": json.loads(store.read_bytes(a.calib_labels))}
    dev_labels = json.loads(store.read_bytes(a.dev_labels)) if store.exists(a.dev_labels) else None
    clf = classifier.fit_classifiers((uids["train"], feats["train"], labels["train"]),
                                     (uids["calibration"], feats["calibration"], labels["calibration"]))
    pred = {r: classifier.predict(clf, uids[r], feats[r]) for r in ("calibration", "development")}

    tensors = {"mean": clf["mean"], "std": clf["std"]}
    for f, mod in clf["models"].items():
        tensors[f"w/{f}"], tensors[f"b/{f}"] = mod["w"], mod["b"].reshape(1)
    clf_path = os.path.join(out_dir, "classifier.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, clf_path)
    feat_sha = hashlib.sha256(open(feat_path, "rb").read()).hexdigest()
    predictions = dict(name=a.name, config=a.config, run_id_of_config=run_job.run_id_for(cfg),
                       vision_stage=cfg.get("vision_stage", {}).get("run_id"), features_sha256=feat_sha,
                       findings=list(clf["models"]), cutoffs={f: mod["cutoff"] for f, mod in clf["models"].items()},
                       calibration=pred["calibration"], development=pred["development"])
    report = dict(name=a.name, config=a.config, features_sha256=feat_sha, n={r: len(uids[r]) for r in ROLES},
                  l2_grid=list(classifier.L2_GRID),
                  findings=classifier.summary(clf, (uids["development"], feats["development"], dev_labels)
                                              if dev_labels else None))
    if dev_labels:
        dev_pred = pred["development"]
        order = list(dev_pred)
        k = len(order) // 2
        shuffled = {u: dev_pred[order[(i + k) % len(order)]] for i, u in enumerate(order)}
        report["label_level_check"] = dict(predicted=metrics.label_level_check(dev_pred, dev_labels),
                                           shuffled=metrics.label_level_check(shuffled, dev_labels),
                                           constant_report=list(metrics.CONSTANT_REPORT_V1))
    files = {}
    for name, obj in (("predictions.json", predictions), ("report.json", report)):
        local = os.path.join(out_dir, name)
        with open(local, "w") as f:
            json.dump(obj, f, indent=1)
        files[f"{prefix}/{name}"] = local
    files[f"{prefix}/classifier.safetensors"] = clf_path
    store.upload_files(files, f"{prefix}: classifier, predictions and report")
    print_report(report)
    return report


def print_report(report):
    def f(x, spec=".2f"):
        return "-" if x is None else format(x, spec)
    print(f"\nfinding classifier '{report['name']}' (logistic regression on pooled image features; studies: "
          + ", ".join(f"{r} {n}" for r, n in report["n"].items()) + ")")
    print(f"{'finding':28s} {'train +':>7s} {'calib AUROC':>11s} {'calib flags ok/wrong':>21s} "
          f"{'dev AUROC':>9s} {'dev flags ok/wrong':>19s}")
    for name, r in report["findings"].items():
        named = r["cutoff"] is not None
        cal = f"{r['correct']}/{r['wrong']}" if named else "never named"
        dev = (f"{r['dev_correct']}/{r['dev_wrong']}" if named else "-") if "dev_auroc" in r else "-"
        print(f"{name:28s} {r['train_positive']:7d} {f(r['calib_auroc']):>11s} {cal:>21s} "
              f"{f(r.get('dev_auroc')):>9s} {dev:>19s}")
    chk = report.get("label_level_check")
    if chk:
        print("\nlabel-level check, development: a report that says exactly the predicted findings (proxy_v1)")
        for kind in ("predicted", "shuffled"):
            c = chk[kind]
            a, k = c["vs_always_normal"], c["vs_constant_report"]
            print(f"  {kind:9s} error {c['error_rate']:.3f} (always-normal {c['always_normal_error_rate']:.3f}, constant "
                  f"{c['constant_report_error_rate']:.3f}) | studies with a finding named {c['studies_with_a_hint']}"
                  f" | vs always-normal {a['baseline_only_errors']} fixed / {a['model_only_errors']} broken, "
                  f"p = {a['p_one_sided']:.3g} | vs constant {k['baseline_only_errors']} / {k['model_only_errors']}, "
                  f"p = {k['p_one_sided']:.3g}")


if __name__ == "__main__":
    main()

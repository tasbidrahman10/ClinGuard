"""Vision check (gate G3, Kaggle session 4): can the base model see chest X-ray findings, and does 4-bit
quantization of its vision tower stop it?

  python scripts/vision_check.py --hf-repo tasbid66/clinguard-runs --name session4 \
      --variants nf4 nf4_visfp16 fp16

For each model variant (untrained base model, all LoRA B = 0):
  - finding probes on the development studies of all sites: AUROC of log P(positive statement) -
    log P(negative statement) between studies whose reference reports the finding and studies whose
    reference rules it out (labels: CheXbert on the reference reports, diagnostics/dev_ref_labels_chexbert.json
    in the store, made in the evaluation environment). 0.5 = the model cannot tell them apart.
  - the image test on one site (true / another study's / blank image).
Writes diagnostics/vision_check_<name>.json (numbers only) locally and to the store.
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.utils import hf_sync  # noqa: E402

LABELS_PATH = "diagnostics/dev_ref_labels_chexbert.json"


def variants():
    import torch
    return {
        "nf4": dict(quant="nf4", compute_dtype=torch.float16),                         # as in sessions 2-3
        "nf4_visfp16": dict(quant="nf4", compute_dtype=torch.float16, vision_fp16=True),
        "fp16": dict(quant="none", compute_dtype=torch.float16, device_map={"": 0}),   # no quantization
        "cpu_fp32": dict(quant="none", compute_dtype=torch.float32),                    # tests (tiny model)
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--variants", nargs="*", default=["nf4", "nf4_visfp16", "fp16"],
                   help="ways of loading the UNTRAINED model (may be empty)")
    p.add_argument("--configs", nargs="*", default=[], help="trained runs to check (their phi_final from the store)")
    p.add_argument("--hf-repo")
    p.add_argument("--store-dir")
    p.add_argument("--partition", default="skew_v1")
    p.add_argument("--image-test-site", default="A")
    p.add_argument("--n", type=int, default=None, help="first n development studies per site only (tests)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pixels", type=int, default=448 * 448)
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--work-dir", default="runs_work")
    p.add_argument("--out-dir", default="diagnostics")
    p.add_argument("--raw-dir", default="data/raw/iu_xray")
    p.add_argument("--images-dir", default="data/raw/iu_xray/images/images_normalized")
    p.add_argument("--manifest-dir", default="manifests")
    a = p.parse_args(argv)
    if bool(a.hf_repo) == bool(a.store_dir):
        p.error("give exactly one of --hf-repo and --store-dir")
    store = hf_sync.HFStore(a.hf_repo) if a.hf_repo else hf_sync.LocalStore(a.store_dir)

    import torch
    from clinguard.data import dataset
    from clinguard.model import diagnostics, load

    model_id, revision = a.model or load.MODEL_ID, a.revision or load.MODEL_REVISION
    m = dataset.verify_source(a.raw_dir, a.manifest_dir, a.partition)
    dev = {s: dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir, a.partition, s, "development")
           for s in m["sites"]}
    if a.n:
        dev = {s: ex[:a.n] for s, ex in dev.items()}
    all_dev = [e for s in sorted(dev) for e in dev[s]]
    labels = json.loads(store.read_bytes(LABELS_PATH))
    os.makedirs(a.work_dir, exist_ok=True)
    blank = os.path.join(a.work_dir, "blank_image.png")
    diagnostics.write_blank_image(blank)
    processor = load.load_processor(model_id, revision, image_pixels=a.pixels)

    results, skipped = {}, []

    def check(label, model, proc, amp, start):
        probes = diagnostics.finding_probe_check(model, proc, all_dev, labels, amp, a.batch)
        img = diagnostics.image_test(model, proc, dev[a.image_test_site], blank, amp, max(1, a.batch // 2))
        results[label] = dict(probes=probes, image_test=img, seconds=round(time.time() - start),
                              peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                              if torch.cuda.is_available() else None)
        print(f"{label} done in {results[label]['seconds']} s", flush=True)

    def free(model):
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name in a.variants:                                  # untrained model, several ways of loading it
        kw = variants()[name]
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = load.load_model(model_id, revision, gradient_checkpointing=False, **kw)
        check(name, model, processor, torch.float16 if kw["compute_dtype"] == torch.float16 else None, start)
        free(model)

    if a.configs:                                            # trained runs: phi_final from the store
        import run_job
        from safetensors.torch import load_file
        from clinguard.model import params
        for path in a.configs:
            cfg = run_job.load_config(path)
            rid = run_job.run_id_for(cfg)
            if not store.exists(f"runs/{rid}/phi_final.safetensors"):
                skipped.append(rid)
                print(f"skip {rid}: no phi_final in the store yet", flush=True)
                continue
            start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            proc, model, amp = run_job.build_model(cfg, store, a.work_dir)
            local = os.path.join(a.work_dir, rid, "phi_final.safetensors")
            if not os.path.exists(local):
                store.download(f"runs/{rid}/phi_final.safetensors", local)
            params.set_phi(params.phi_params(model), load_file(local)["phi"])
            check(rid, model, proc, amp, start)
            free(model)

    report = dict(name=a.name, partition=a.partition, model=model_id, revision=revision, pixels=a.pixels,
                  statements=diagnostics.PROBE_STATEMENTS, n_dev=len(all_dev), image_test_site=a.image_test_site,
                  results=results, skipped=skipped)
    if not results:
        raise SystemExit("nothing to check: no variant given and no config has a phi_final yet")
    os.makedirs(a.out_dir, exist_ok=True)
    out_local = os.path.join(a.out_dir, f"vision_check_{a.name}.json")
    with open(out_local, "w") as f:
        json.dump(report, f, indent=1)
    store.upload_files({f"diagnostics/vision_check_{a.name}.json": out_local}, f"vision check {a.name}")
    print_report(report)
    return report


def print_report(report):
    def f(x, spec=".2f"):
        return format(x, spec) if x is not None else "-"
    findings = list(report["statements"])
    width = max([16] + [len(n) for n in report["results"]])
    print(f"\nvision check: can the model tell development studies with a finding from studies without it?")
    print("AUROC of the model's preference for the positive statement (0.5 = cannot tell; 1.0 = perfect)")
    first = next(iter(report["results"].values()))["probes"]
    print(f"{'model':{width}s} " + " ".join(f"{c[:16]:>16s}" for c in findings) + " | image test site "
          f"{report['image_test_site']}: abnormal, true image better")
    print(f"{'(present/absent)':{width}s} " + " ".join(
        f"{str(first[c]['n_present']) + '/' + str(first[c]['n_absent']):>16s}" for c in findings))
    for name, r in report["results"].items():
        ab = r["image_test"]["abnormal"]
        print(f"{name:{width}s} " + " ".join(f"{f(r['probes'][c]['auroc']):>16s}" for c in findings)
              + f" | {f(ab.get('share_true_image_better'), '.0%')} of studies   ({r['seconds']} s, peak {r['peak_gpu_gb']} GB)")
    for rid in report.get("skipped", []):
        print(f"{rid}: not checked (no phi_final in the store yet)")


if __name__ == "__main__":
    main()

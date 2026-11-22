"""Image test (gate G3): does each trained model use the X-ray? See clinguard/model/diagnostics.py.

  python scripts/image_test.py --hf-repo tasbid66/clinguard-runs --name session3 \
      --configs configs/experiment/s1_single_site_A.yaml configs/experiment/g3_A_lr2e4.yaml ...

For every config whose run has a phi_final in the store, and once for the base model (all B = 0), computes
the reference-report loss on one site's development studies with the true image, another study's image and
a blank image. Writes diagnostics/image_test_<name>.json (numbers only) and uploads it to the same path in
the store. The base model is loaded once per LoRA target set.
"""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import run_job  # noqa: E402
from clinguard.utils import hf_sync  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--hf-repo")
    p.add_argument("--store-dir")
    p.add_argument("--site", default="A")
    p.add_argument("--role", default="development", choices=("development", "calibration"))
    p.add_argument("--n", type=int, default=None, help="first n studies only (tests)")
    p.add_argument("--batch", type=int, default=4)
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
    from safetensors.torch import load_file
    from clinguard.data import dataset
    from clinguard.model import diagnostics, lora, params

    cfgs = [(path, run_job.load_config(path)) for path in a.configs]
    partition = cfgs[0][1]["partition"]
    if any(c["partition"] != partition for _, c in cfgs):
        raise SystemExit("all configs must use the same partition")
    dataset.verify_source(a.raw_dir, a.manifest_dir, partition)
    examples = dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir, partition, a.site, a.role)
    examples = examples[:a.n] if a.n else examples
    os.makedirs(a.work_dir, exist_ok=True)
    blank = os.path.join(a.work_dir, "blank_image.png")
    diagnostics.write_blank_image(blank)

    groups = {}
    for path, cfg in cfgs:
        key = (tuple(cfg["lora"].get("targets", lora.TARGETS)), cfg["lora"]["a_file"], cfg["lora"]["a_sha256"],
               json.dumps(cfg["model"], sort_keys=True))
        groups.setdefault(key, []).append((path, cfg))

    results, skipped = {}, []
    for members in groups.values():
        ready = []
        for path, cfg in members:
            rid = run_job.run_id_for(cfg)
            if store.exists(f"runs/{rid}/phi_final.safetensors"):
                ready.append((rid, cfg))
            else:
                skipped.append(rid)
                print(f"skip {rid}: no phi_final in the store yet", flush=True)
        if not ready and results:
            continue
        processor, model, amp = run_job.build_model(members[0][1], store, a.work_dir)
        b_params = params.phi_params(model)
        if "base" not in results:
            params.set_phi(b_params, torch.zeros(sum(q.numel() for q in b_params)))
            results["base"] = diagnostics.image_test(model, processor, examples, blank, amp, a.batch)
            print("base done", flush=True)
        for rid, cfg in ready:
            local = os.path.join(a.work_dir, rid, "phi_final.safetensors")
            if not os.path.exists(local):
                store.download(f"runs/{rid}/phi_final.safetensors", local)
            params.set_phi(b_params, load_file(local)["phi"])
            results[rid] = diagnostics.image_test(model, processor, examples, blank, amp, a.batch)
            print(f"{rid} done", flush=True)
        del model, b_params
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = dict(name=a.name, partition=partition, site=a.site, role=a.role, n=len(examples),
                  conditions=list(diagnostics.CONDITIONS), results=results, skipped=skipped,
                  code_commit=run_job.git_commit())
    os.makedirs(a.out_dir, exist_ok=True)
    out_local = os.path.join(a.out_dir, f"image_test_{a.name}.json")
    with open(out_local, "w") as f:
        json.dump(report, f, indent=1)
    store.upload_files({f"diagnostics/image_test_{a.name}.json": out_local}, f"image test {a.name}")

    print(f"\nimage test, site {a.site} {a.role}, {len(examples)} studies; loss of the reference report "
          f"(lower = better predicted)")
    print(f"{'model':58s} {'true':>6s} {'other':>6s} {'blank':>6s} | abnormal: {'true':>6s} {'other':>6s} "
          f"true-better")
    def f(x, spec="6.3f"):
        return format(x, spec) if x is not None else "     -"
    for label, r in results.items():
        al, ab = r["all"], r["abnormal"]
        print(f"{label[:58]:58s} {f(al['loss_true'])} {f(al['loss_shuffled'])} {f(al['loss_blank'])} | "
              f"{'':9s} {f(ab['loss_true'])} {f(ab['loss_shuffled'])} {f(ab.get('share_true_image_better'), '6.0%')}")
    return report


if __name__ == "__main__":
    main()

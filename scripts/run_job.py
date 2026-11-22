"""Run (or resume) one experiment from a YAML config, with the shared HF repo as the source of truth
(plan section 4.1).

  python scripts/run_job.py --config configs/experiment/s1_single_site_A.yaml --hf-repo tasbid66/clinguard-runs

Steps: compute the run id from the config -> take the lease (refuse if another live session holds it or the
run is finished) -> download the run folder if it exists and refuse a different code commit -> load data,
model, LoRA (shared A verified by checksum) and sites -> run the DiLoCo engine, uploading the run folder and
renewing the lease after every round -> save phi_final -> generate on the development split -> finish.
Rerunning the same command after a crash continues where the store left off.

Zero-shot is a config with total_steps_per_site: 0. Local-only training is a run with one site, eta_out 1
and mu 0 (the outer step then returns the site's own parameters).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clinguard.utils import hf_sync  # noqa: E402
from clinguard.utils.hashing import hash_config  # noqa: E402

RUN_FILES = ("checkpoint.pt", "run.json", "rounds.jsonl")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("method", "partition", "sites", "seed", "model", "engine", "lora", "eval"):
        if key not in cfg:
            raise ValueError(f"{path}: missing '{key}'")
    return cfg


def run_id_for(cfg):
    e = cfg["engine"]
    return f"{cfg['method']}_{cfg['partition']}_H{e['H']}_s{cfg['seed']}_{hash_config(cfg)[:8]}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:                                                   # noqa: BLE001
        return None


def build_model(cfg, store, work_dir):
    """(processor, model, amp_dtype) for a run config: base model, LoRA on cfg["lora"]["targets"] (default
    q/k/v/o), and the shared A matrices downloaded from the store and checked against their SHA-256."""
    import torch
    from clinguard.model import load, lora
    mc = cfg["model"]
    quant = mc.get("quant", "nf4")
    dtype = torch.float16 if quant == "nf4" else torch.float32
    processor = load.load_processor(mc.get("name", load.MODEL_ID), mc.get("revision", load.MODEL_REVISION),
                                    image_pixels=mc["pixels"])
    model = load.load_model(mc.get("name", load.MODEL_ID), mc.get("revision", load.MODEL_REVISION), quant=quant,
                            compute_dtype=dtype, kbit_prepare=mc.get("kbit_prepare", False))
    model = lora.apply_fixed_basis_lora(model, targets=tuple(cfg["lora"].get("targets", lora.TARGETS)))
    a_local = os.path.join(work_dir, "shared", os.path.basename(cfg["lora"]["a_file"]))
    if not os.path.exists(a_local):
        store.download(cfg["lora"]["a_file"], a_local)
    lora.load_lora_a(model, a_local, cfg["lora"]["a_sha256"])
    return processor, model, (torch.float16 if quant == "nf4" else None)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf-repo", help="private HF model repo (token from HF_TOKEN)")
    p.add_argument("--store-dir", help="local folder standing in for the HF repo (tests)")
    p.add_argument("--account", default=None, help="who is running (default: the HF user name)")
    p.add_argument("--takeover", action="store_true", help="take an unexpired lease from a session known to be dead")
    p.add_argument("--work-dir", default="runs_work")
    p.add_argument("--raw-dir", default="data/raw/iu_xray")
    p.add_argument("--images-dir", default="data/raw/iu_xray/images/images_normalized")
    p.add_argument("--manifest-dir", default="manifests")
    p.add_argument("--stop-after-round", type=int, default=None, help="stop early (resume tests)")
    p.add_argument("--upload-every-min", type=float, default=15.0,
                   help="upload the run folder at most this often (plus the final round and a deliberate stop)")
    a = p.parse_args(argv)
    if bool(a.hf_repo) == bool(a.store_dir):
        p.error("give exactly one of --hf-repo and --store-dir")

    cfg = load_config(a.config)
    run_id = run_id_for(cfg)
    store = hf_sync.HFStore(a.hf_repo) if a.hf_repo else hf_sync.LocalStore(a.store_dir)
    account = a.account or (store.api.whoami()["name"] if a.hf_repo else "local")
    commit = git_commit()
    remote = f"runs/{run_id}"
    print(f"run {run_id} | account {account} | commit {commit}", flush=True)

    lease = hf_sync.Lease(store, run_id, hf_sync.holder_identity(account), commit)
    try:
        lease.acquire(takeover=a.takeover)
    except hf_sync.RunFinished:
        print(f"{run_id} is already finished; nothing to do")
        return run_id

    out_dir = os.path.join(a.work_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    remote_meta = hf_sync.read_json(store, f"{remote}/run.json")
    if remote_meta is not None:
        if remote_meta.get("code_commit") != commit:
            raise SystemExit(f"refusing to resume {run_id}: it was started at commit {remote_meta.get('code_commit')}, "
                             f"this checkout is {commit}")
        for name in RUN_FILES:
            if store.exists(f"{remote}/{name}"):
                store.download(f"{remote}/{name}", os.path.join(out_dir, name))
        print("resuming from the store", flush=True)

    # Heavy imports only after the cheap checks above.
    from clinguard.data import dataset
    from clinguard.engine.runner import RunConfig, run
    from clinguard.model import params
    from clinguard.model.generate import generate_findings
    from clinguard.model.site import VLMSite
    from safetensors.torch import save_file

    m = dataset.verify_source(a.raw_dir, a.manifest_dir, cfg["partition"])
    mc, lim = cfg["model"], cfg.get("limits", {})
    processor, model, amp = build_model(cfg, store, a.work_dir)

    def examples(site, role, key):
        ex = dataset.load_examples(a.raw_dir, a.images_dir, a.manifest_dir, cfg["partition"], site, role)
        return ex[:lim[key]] if key in lim else ex

    sites = [VLMSite(s, model, processor, examples(s, "train", "train"), examples(s, "calibration", "calib"),
                     micro_batch=mc["micro_batch"], grad_accum=mc["grad_accum"], lr=mc["lr"],
                     warmup_steps=mc["warmup_steps"], seed=cfg["seed"] * 1000 + i, amp_dtype=amp)
             for i, s in enumerate(cfg["sites"])]
    engine = RunConfig(**{**cfg["engine"], "seed": cfg["seed"], "method": cfg["method"]})
    extra = {k: v for k, v in cfg.items() if k not in ("engine", "seed", "method")}

    last_upload = [time.monotonic()]

    def upload(t, folder):
        """After round t: upload the run folder (one commit, together with the lease renewal) when the last
        upload is at least --upload-every-min old, at the final round, or at a deliberate stop. A crash
        therefore loses at most that much compute, while short rounds do not flood the repo with commits."""
        due = (time.monotonic() - last_upload[0] >= 60 * a.upload_every_min or t + 1 == engine.rounds
               or t == a.stop_after_round)
        if not due:
            return
        files = {f"{remote}/{n}": os.path.join(folder, n) for n in RUN_FILES if os.path.exists(os.path.join(folder, n))}
        lease.renew(files, f"{run_id}: round {t} complete")
        last_upload[0] = time.monotonic()
        print(f"round {t} uploaded", flush=True)

    phi0 = params.get_phi(params.phi_params(model))
    phi, _ = run(engine, sites, phi0, out_dir, manifest_hash=m["manifest_hash"], extra_config=extra,
                 stop_after_round=a.stop_after_round, on_round_end=upload)
    if engine.rounds == 0:                                   # zero-shot: record run.json once
        lease.renew({f"{remote}/run.json": os.path.join(out_dir, "run.json")}, f"{run_id}: run.json")
    if a.stop_after_round is not None and a.stop_after_round + 1 < engine.rounds:
        lease.pause()
        print(f"stopped after round {a.stop_after_round} as asked; lease handed back (status paused)")
        return run_id

    if not store.exists(f"{remote}/phi_final.safetensors"):
        save_file({"phi": phi.contiguous()}, os.path.join(out_dir, "phi_final.safetensors"))
        lease.renew({f"{remote}/phi_final.safetensors": os.path.join(out_dir, "phi_final.safetensors")},
                    f"{run_id}: final phi")
    params.set_phi(params.phi_params(model), phi)
    ev = cfg["eval"]
    for site in ev.get("dev_sites", []):
        path = f"{remote}/generations/dev_{site}.jsonl"
        if store.exists(path):
            continue
        dev = examples(site, "development", "dev")
        texts = generate_findings(model, processor, dev, batch_size=ev.get("gen_batch", 8),
                                  max_new_tokens=ev.get("max_new_tokens", 160))
        local = os.path.join(out_dir, "generations", f"dev_{site}.jsonl")
        os.makedirs(os.path.dirname(local), exist_ok=True)
        write_jsonl(local, [dict(run_id=run_id, partition=cfg["partition"], site=site, uid=e.uid, group=e.group,
                                 needs_prior=e.needs_prior, generated=t, reference=e.findings)
                            for e, t in zip(dev, texts)])
        lease.renew({path: local}, f"{run_id}: development generations, site {site}")
        print(f"generated {len(dev)} development reports for site {site}", flush=True)
    lease.finish()
    print(f"{run_id} finished", flush=True)
    return run_id


if __name__ == "__main__":
    main()

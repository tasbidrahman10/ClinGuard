"""Shared-compute protocol (plan section 4.1): the lease, and scripts/run_job.py end to end against a local
folder standing in for the HF repo (tiny Qwen2.5-VL, real images, CPU)."""
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import torch
import yaml

from clinguard.utils import hf_sync
from clinguard.utils.hf_sync import Conflict, Lease, LeaseHeld, LocalStore, RunFinished

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def holder(name):
    return dict(account=name, session=f"{name}-session")


# ------------------------------------------------------------------------------------------- lease

def test_lease_blocks_a_second_live_session(tmp_path):
    store = LocalStore(str(tmp_path))
    mine, theirs = Lease(store, "r1", holder("tazvir"), "c1"), Lease(store, "r1", holder("puja"), "c1")
    mine.acquire()
    with pytest.raises(LeaseHeld):
        theirs.acquire()
    mine.renew()
    with pytest.raises(LeaseHeld):
        theirs.renew()                                              # a non-holder can never renew
    rec = theirs.acquire(takeover=True)                             # explicit takeover of a dead session
    assert rec["holder"]["account"] == "puja" and rec["previous_holder"]["account"] == "tazvir"
    with pytest.raises(LeaseHeld):
        mine.renew()                                                # the old session learns it lost the run


def test_expired_and_paused_leases_can_be_taken_finished_runs_cannot(tmp_path):
    store = LocalStore(str(tmp_path))
    first = Lease(store, "r2", holder("a"), "c1")
    first.acquire()
    rec = json.loads(store.read_bytes("runs/r2/LEASE.json"))
    rec["expires"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store.write_bytes("runs/r2/LEASE.json", json.dumps(rec).encode(), "expire")
    Lease(store, "r2", holder("b"), "c1").acquire()                 # expired: allowed without takeover
    second = Lease(store, "r2", holder("b"), "c1")
    second.holder = json.loads(store.read_bytes("runs/r2/LEASE.json"))["holder"]
    second.pause()
    Lease(store, "r2", holder("c"), "c1").acquire()                 # paused: allowed at once
    third = Lease(store, "r2", holder("c"), "c1")
    third.finish()
    with pytest.raises(RunFinished):
        Lease(store, "r2", holder("d"), "c1").acquire(takeover=True)


def test_lease_write_retries_after_a_concurrent_commit(tmp_path):
    class Racy(LocalStore):
        raced = False

        def commit(self, adds, message, parent=None):
            if not self.raced and parent is not None:                 # someone commits in between, once
                self.raced = True
                super().commit({"runs/other/x": b"x"}, "other")
            return super().commit(adds, message, parent)

    store = Racy(str(tmp_path))
    rec = Lease(store, "r3", holder("a"), "c1").acquire()
    assert store.raced and rec["status"] == "running"
    with pytest.raises(Conflict):
        store.write_bytes("runs/r3/y", b"y", "stale", parent="0")     # a stale parent is refused


# ------------------------------------------------------------------------------------ run_job end to end

def _run_job():
    spec = importlib.util.spec_from_file_location("run_job", os.path.join(ROOT, "scripts", "run_job.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, tiny, method):
    from clinguard.model import load, lora
    fresh, _ = tiny
    os.makedirs(tmp_path, exist_ok=True)
    store_dir = tmp_path / "store"
    a_path = tmp_path / "a.safetensors"
    sha = lora.save_lora_a(lora.apply_fixed_basis_lora(fresh()), str(a_path))
    LocalStore(str(store_dir)).upload_files({"shared/lora_A.safetensors": str(a_path)}, "A")
    cfg = dict(method=method, partition="skew_v1", sites=["A", "B"], seed=11,
               model=dict(name=load.TINY_MODEL_ID, revision=load.TINY_MODEL_REVISION, pixels=112 * 112,
                          micro_batch=1, grad_accum=2, lr=1e-3, warmup_steps=2, quant="none"),
               engine=dict(H=1, total_steps_per_site=3, eta_out=0.7, mu=0.9, weighting="fixed_q"),
               lora=dict(a_file="shared/lora_A.safetensors", a_sha256=sha),
               eval=dict(dev_sites=["A"], gen_batch=2, max_new_tokens=4),
               limits=dict(train=4, calib=2, dev=2))
    path = tmp_path / f"{method}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path), str(store_dir)


def _phi(store_dir, run_id):
    from safetensors.torch import load_file
    return load_file(os.path.join(store_dir, "runs", run_id, "phi_final.safetensors"))["phi"]


def test_run_job_finishes_resumes_from_the_store_and_refuses_reruns(tmp_path, tiny, iu_data, monkeypatch):
    monkeypatch.chdir(ROOT)
    rj = _run_job()
    straight_cfg, store_dir = _setup(tmp_path, tiny, "straight")
    common = ["--store-dir", store_dir, "--work-dir", str(tmp_path / "work1")]
    rid = rj.main(["--config", straight_cfg] + common)
    store = LocalStore(store_dir)
    assert json.loads(store.read_bytes(f"runs/{rid}/LEASE.json"))["status"] == "finished"
    rows = [json.loads(l) for l in store.read_bytes(f"runs/{rid}/generations/dev_A.jsonl").decode().splitlines()]
    assert len(rows) == 2 and all(r["site"] == "A" and "generated" in r and r["reference"] for r in rows)
    assert len(store.read_bytes(f"runs/{rid}/rounds.jsonl").decode().splitlines()) == 3
    assert rj.main(["--config", straight_cfg] + common) == rid       # finished: nothing to do

    # Same training, interrupted after round 0 and resumed by a "new session" with an empty work folder.
    split_cfg, _ = _setup(tmp_path / "s", tiny, "split")
    split_store = str(tmp_path / "s" / "store")
    rid2 = rj.main(["--config", split_cfg, "--store-dir", split_store, "--work-dir", str(tmp_path / "w2"),
                    "--stop-after-round", "0"])
    assert json.loads(LocalStore(split_store).read_bytes(f"runs/{rid2}/LEASE.json"))["status"] == "paused"
    rj.main(["--config", split_cfg, "--store-dir", split_store, "--work-dir", str(tmp_path / "w3"),
             "--account", "teammate"])
    assert torch.equal(_phi(store_dir, rid), _phi(split_store, rid2))      # bit for bit on CPU


def test_run_job_refuses_to_resume_on_different_code(tmp_path, tiny, iu_data, monkeypatch):
    monkeypatch.chdir(ROOT)
    rj = _run_job()
    cfg, store_dir = _setup(tmp_path, tiny, "commitcheck")
    rid = rj.main(["--config", cfg, "--store-dir", store_dir, "--work-dir", str(tmp_path / "w"), "--stop-after-round", "0"])
    store = LocalStore(store_dir)
    meta = json.loads(store.read_bytes(f"runs/{rid}/run.json"))
    meta["code_commit"] = "0" * 40
    store.write_bytes(f"runs/{rid}/run.json", json.dumps(meta).encode(), "tamper")
    with pytest.raises(SystemExit, match="refusing to resume"):
        rj.main(["--config", cfg, "--store-dir", store_dir, "--work-dir", str(tmp_path / "w4")])

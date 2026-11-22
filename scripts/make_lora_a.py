"""Make the shared LoRA A file for a wider target set (decision I28). Runs on a laptop: the model is built on
the meta device from its config, so no weights are downloaded and no GPU is needed.

  python scripts/make_lora_a.py --hf-repo tasbid66/clinguard-runs \
      --targets q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --out shared/lora_A_r16_qkvo_mlp_seed2026.safetensors --upload

The q/k/v/o matrices are copied unchanged from the primary A file (checked against its SHA-256), so a run
with the wider set differs from a q/k/v/o run only by the added modules. New matrices follow peft's rule
(uniform in +-1/sqrt(in_features)), one seeded generator per module name. Prints the new file's SHA-256 for
the run configs. Refuses to replace a different file already in the store.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

PRIMARY_A = "shared/lora_A_r16_seed2026.safetensors"
PRIMARY_A_SHA = "9d927008b45ce51c3fc190f4dca23059290a9b2beaac43d16dccd9c2e0dbef56"


def skeleton(model_id, revision):
    """The model's module tree on the meta device (shapes and names only)."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(config)


def main(argv=None):
    from safetensors.torch import load_file, save_file

    from clinguard.model import load, lora
    from clinguard.utils.hashing import sha256_file
    from clinguard.utils.hf_sync import HFStore, LocalStore

    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo")
    p.add_argument("--store-dir", help="local folder standing in for the HF repo (tests)")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--out", required=True, help="path inside the store, e.g. shared/lora_A_r16_qkvo_mlp_seed2026.safetensors")
    p.add_argument("--base", default=PRIMARY_A)
    p.add_argument("--base-sha256", default=PRIMARY_A_SHA)
    p.add_argument("--model", default=load.MODEL_ID)
    p.add_argument("--revision", default=load.MODEL_REVISION)
    p.add_argument("--work-dir", default="runs_work")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args(argv)
    if bool(a.hf_repo) == bool(a.store_dir):
        p.error("give exactly one of --hf-repo and --store-dir")
    store = HFStore(a.hf_repo) if a.hf_repo else LocalStore(a.store_dir)

    base_local = os.path.join(a.work_dir, "shared", os.path.basename(a.base))
    if not os.path.exists(base_local):
        store.download(a.base, base_local)
    if sha256_file(base_local) != a.base_sha256:
        raise SystemExit(f"{a.base}: checksum does not match {a.base_sha256[:12]}...")
    base = load_file(base_local)

    model = skeleton(a.model, a.revision)
    specs = lora.lora_a_specs(model, targets=tuple(a.targets))
    state = lora.extend_lora_a(base, specs)
    out_local = os.path.join(a.work_dir, "shared", os.path.basename(a.out))
    # No safetensors metadata: it is written from an unordered map, so the bytes (and the SHA-256) would
    # change from build to build. Without it the file is byte-reproducible; provenance is in decision I28.
    save_file(state, out_local)
    sha = sha256_file(out_local)
    modules = dict(model.named_modules())
    phi_size = sum(lora.RANK * modules[n].out_features for n in lora.target_module_names(model, tuple(a.targets)))
    print(f"{len(state)} A matrices ({len(base)} copied from {a.base}); phi size {phi_size:,}; sha256 {sha}")

    if a.upload:
        if store.exists(a.out):
            remote = os.path.join(a.work_dir, "shared", "remote_" + os.path.basename(a.out))
            store.download(a.out, remote)
            if sha256_file(remote) != sha:
                raise SystemExit(f"{a.out} already exists in the store with different contents; not replaced")
            print(f"{a.out} is already in the store (same contents)")
        else:
            store.upload_files({a.out: out_local}, f"shared LoRA A file for targets {','.join(a.targets)}")
            print(f"uploaded {a.out}")
    return sha, phi_size


if __name__ == "__main__":
    main()

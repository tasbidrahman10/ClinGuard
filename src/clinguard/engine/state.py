"""Round-level checkpoints. A checkpoint is written atomically after every complete round and holds
everything needed to continue exactly as an uninterrupted run would (test A.7).
"""
import os

import torch

CKPT_NAME = "checkpoint.pt"


def save_checkpoint(ckpt_dir, *, next_round, phi, m, sites, accountant, config_hash, manifest_hash):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = dict(
        next_round=next_round, phi=phi, m=m,
        sites={s.name: s.state_dict() for s in sites},
        accountant=accountant.state(),
        rng_torch=torch.get_rng_state(),
        config_hash=config_hash, manifest_hash=manifest_hash,
    )
    path = os.path.join(ckpt_dir, CKPT_NAME)
    tmp = path + ".part"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)          # atomic: a crash leaves either the old or the new file
    return path


def load_checkpoint(ckpt_dir, *, sites, accountant, config_hash, manifest_hash):
    """Returns (next_round, phi, m), or None when there is nothing to resume."""
    path = os.path.join(ckpt_dir, CKPT_NAME)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # our own file, not untrusted input
    if ckpt["config_hash"] != config_hash:
        raise RuntimeError("checkpoint was written with a different config; refusing to resume")
    if ckpt["manifest_hash"] != manifest_hash:
        raise RuntimeError("checkpoint was written with a different data manifest; refusing to resume")
    if set(ckpt["sites"]) != {s.name for s in sites}:
        raise RuntimeError("checkpoint sites do not match this run's sites")
    for s in sites:
        s.load_state_dict(ckpt["sites"][s.name])
    accountant.load(ckpt["accountant"])
    torch.set_rng_state(ckpt["rng_torch"])
    return ckpt["next_round"], ckpt["phi"], ckpt["m"]

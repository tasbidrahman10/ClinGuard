"""Hashes used to tie a run to its exact config, data manifest and files."""
import hashlib
import json


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_config(cfg):
    """Order-independent hash of a JSON-serialisable config."""
    return sha256_text(json.dumps(cfg, sort_keys=True, separators=(",", ":")))

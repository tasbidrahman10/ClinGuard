"""Split manifests: which study belongs to which site and data role.

Layout:  <manifest_dir>/<partition>/<site>_<role>.csv   columns: uid, image, group, needs_prior
         <manifest_dir>/<partition>/manifest.json       file hashes, versions, seeds, counts

The internal test split cannot be loaded before the freeze (plan section 0, rule 5).
"""
import csv
import json
import os
from collections import Counter

from clinguard.data.partition import ROLES
from clinguard.utils.hashing import sha256_file, sha256_text

COLUMNS = ["uid", "image", "group", "needs_prior"]


def write_partition(manifest_dir, name, rows, assignment, sites, meta):
    """rows: {uid: dict(image, group, needs_prior)}; assignment: {uid: (site, role)}."""
    out_dir = os.path.join(manifest_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    files, counts = {}, {}
    for site in sites:
        for role in ROLES:
            uids = sorted((u for u, sr in assignment.items() if sr == (site, role)), key=int)
            fname = f"{site}_{role}.csv"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(COLUMNS)
                for u in uids:
                    w.writerow([u, rows[u]["image"], rows[u]["group"], int(rows[u]["needs_prior"])])
            files[fname] = sha256_file(os.path.join(out_dir, fname))
            counts[f"{site}_{role}"] = dict(sorted(Counter(rows[u]["group"] for u in uids).items()),
                                            total=len(uids))
    manifest = dict(partition=name, sites=list(sites), roles=ROLES, files=files, counts=counts,
                    n_studies=len(assignment), **meta)
    manifest["manifest_hash"] = sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(files.items())))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest


def read_manifest(manifest_dir, name):
    with open(os.path.join(manifest_dir, name, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_split(manifest_dir, name, site, role, unlock_test=False, freeze_tag=None):
    """Rows of one site x role. The test split needs unlock_test=True and the freeze tag."""
    if role not in ROLES:
        raise ValueError(f"unknown role '{role}'")
    if role == "test" and not (unlock_test and freeze_tag):
        raise PermissionError("the internal test split is locked until the freeze; "
                              "pass --unlock-test together with the freeze tag")
    return _read_rows(os.path.join(manifest_dir, name, f"{site}_{role}.csv"))


def verify_partition(manifest_dir, name):
    """Raises if a file changed, a study appears twice, or counts disagree. Returns {uid: (site, role)}."""
    m = read_manifest(manifest_dir, name)
    seen = {}
    for site in m["sites"]:
        for role in m["roles"]:
            fname = f"{site}_{role}.csv"
            path = os.path.join(manifest_dir, name, fname)
            if sha256_file(path) != m["files"][fname]:
                raise ValueError(f"{name}/{fname} does not match its recorded hash")
            rows = _read_rows(path)
            if len(rows) != m["counts"][f"{site}_{role}"]["total"]:
                raise ValueError(f"{name}/{fname}: row count differs from manifest.json")
            for r in rows:
                if r["uid"] in seen:
                    raise ValueError(f"uid {r['uid']} is in both {seen[r['uid']]} and {(site, role)}")
                seen[r["uid"]] = (site, role)
    if len(seen) != m["n_studies"]:
        raise ValueError(f"{name}: {len(seen)} studies on disk, {m['n_studies']} recorded")
    return seen

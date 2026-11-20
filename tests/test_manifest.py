"""Integrity of the committed split manifests (plan section 5.6; part of A.7).

Checks every file against its recorded hash, that no study appears in two roles or two sites, that
counts match, and that a study has the same data role in both partitions.
"""
import os

import pytest

from clinguard.data import manifest

MANIFESTS = "manifests"
PARTITIONS = ["skew_v1", "iid_v1"]

# Frozen at gate G1 on 2026-11-20. These values never change; a failure here means the data changed.
FROZEN_HASHES = {
    "skew_v1": "fba077ca06499e734d2dcb3608b4c0a0630c05c3ced74392fb42694aa9a9a4a5",
    "iid_v1": "77e5cd83bdf165f87f73c81543fa59564c9f2d126223bee45cdff5a5bb4b2348",
}

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(MANIFESTS, p, "manifest.json")) for p in PARTITIONS),
    reason="manifests not generated yet (run scripts/prepare_data.py)")


@pytest.mark.parametrize("name", PARTITIONS)
def test_partition_is_intact_with_no_overlap(name):
    seen = manifest.verify_partition(MANIFESTS, name)
    m = manifest.read_manifest(MANIFESTS, name)
    assert len(seen) == m["n_studies"] > 0
    assert sum(c["total"] for c in m["counts"].values()) == m["n_studies"]


def test_roles_agree_across_partitions():
    skew = manifest.verify_partition(MANIFESTS, "skew_v1")
    iid = manifest.verify_partition(MANIFESTS, "iid_v1")
    assert set(skew) == set(iid)
    assert all(skew[u][1] == iid[u][1] for u in skew)


def test_partitions_share_source_and_versions():
    a, b = (manifest.read_manifest(MANIFESTS, p) for p in PARTITIONS)
    for key in ("raw_sha256", "clean_version", "prior_version", "groups_version", "split_seed",
                "cleaned_text_sha256"):
        assert a[key] == b[key]


@pytest.mark.parametrize("name", PARTITIONS)
def test_manifests_are_frozen_and_unchanged(name):
    m = manifest.read_manifest(MANIFESTS, name)
    assert m["status"] == "frozen" and m["image_check"] == "passed"
    assert (m["clean_version"], m["prior_version"], m["groups_version"], m["split_seed"]) == \
        ("clean_v1", "prior_v2", "groups_v1", 20261110)
    assert m["manifest_hash"] == FROZEN_HASHES[name]

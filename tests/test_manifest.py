"""Integrity of the committed split manifests (plan section 5.6; part of A.7).

Checks every file against its recorded hash, that no study appears in two roles or two sites, that
counts match, and that a study has the same data role in both partitions.
"""
import os

import pytest

from clinguard.data import manifest

MANIFESTS = "manifests"
PARTITIONS = ["skew_v1", "iid_v1"]

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
    for key in ("raw_sha256", "clean_version", "groups_version", "split_seed", "cleaned_text_sha256"):
        assert a[key] == b[key]

"""Cleaning, group labels, partitions and manifests, on small synthetic inputs (no IU data needed)."""
import csv
from collections import Counter

import pytest
import yaml

from clinguard.data import clean, labels, manifest, partition, prepare
from clinguard.data.ingest import PROJECTION_COLUMNS, REPORT_COLUMNS

GROUPS = labels.load_groups("configs/data/groups_v1.yaml")
SKEW = yaml.safe_load(open("configs/data/skew_v1.yaml"))
IID = yaml.safe_load(open("configs/data/iid_v1.yaml"))


# ---------------------------------------------------------------- cleaning

def test_masked_tokens_are_deleted_and_spacing_tidied():
    assert clean.clean_findings("Midline sternotomy XXXX.  Clear   lungs.") == "Midline sternotomy. Clear lungs."
    assert clean.clean_findings("XXXX XXXX opacities compatible with atelectasis.") == \
        "opacities compatible with atelectasis."
    assert clean.clean_findings("No effusion, XXXX, or pneumothorax.") == "No effusion, or pneumothorax."


def test_mostly_masked_sentences_are_removed():
    assert clean.clean_findings("Borderline cardiomegaly. Inferior XXXX XXXX XXXX. Clear lungs.") == \
        "Borderline cardiomegaly. Clear lungs."
    assert clean.clean_findings("XXXX XXXX.") == ""
    assert clean.clean_findings("Heart XXXX.") == "Heart."            # exactly half is kept


def test_casing_punctuation_and_decimals_survive():
    text = "Cardiothoracic ratio of 16.2/24.7. No Pneumothorax; lungs are clear."
    assert clean.clean_findings(text) == text


def test_needs_prior_uses_whole_words_only():
    assert clean.needs_prior("Heart size is stable.")
    assert clean.needs_prior("Unchanged from PRIOR exam.")
    assert not clean.needs_prior("Priority follow-up is not required. Stablemate.")
    assert not clean.needs_prior("The lungs are clear.")


# ---------------------------------------------------------------- group labels

@pytest.mark.parametrize("mesh,group", [
    ("normal", "normal"),
    ("Cardiomegaly/borderline;Pulmonary Artery/enlarged", "cardio_pleural"),
    ("Cardiac Shadow/enlarged/mild", "cardio_pleural"),
    ("Cardiac Shadow/borderline", "other_abnormal"),                 # attribute required
    ("Aorta, Thoracic/tortuous/mild", "cardio_pleural"),
    ("Aorta/prominent", "other_abnormal"),
    ("Opacity/lung/base/left;Cardiomegaly/mild", "cardio_pleural"),  # first group in the order wins
    ("Calcified Granuloma/lung/upper lobe/right", "parenchymal"),
    ("Subcutaneous Emphysema", "other_abnormal"),                    # not lung emphysema
    ("Thoracic Vertebrae/degenerative;Spine/degenerative", "other_abnormal"),
    ("No Indexing", "other_abnormal"),
])
def test_assign_group(mesh, group):
    assert labels.assign_group(mesh, GROUPS) == group


# ---------------------------------------------------------------- partitions

def synthetic(n=1200):
    groups = ["normal", "cardio_pleural", "parenchymal", "other_abnormal"]
    return [(str(i + 1), groups[i % 4]) for i in range(n)]


def test_skew_partition_is_deterministic_and_complete():
    a = partition.skew_partition(synthetic(), SKEW)
    assert a == partition.skew_partition(reversed(synthetic()), SKEW)       # input order is irrelevant
    assert len(a) == 1200 and {s for s, _ in a.values()} == {"A", "B", "C"}


def test_a_study_keeps_its_site_when_others_are_removed():
    full = partition.skew_partition(synthetic(), SKEW)
    fewer = partition.skew_partition(synthetic()[:900], SKEW)
    assert all(fewer[u][0] == full[u][0] for u in fewer)


def test_skew_follows_the_table_and_role_fractions():
    studies = synthetic(6000)
    a = partition.skew_partition(studies, SKEW)
    group_of = dict(studies)
    site_given_group = Counter((group_of[u], s) for u, (s, _) in a.items())
    for g, probs in SKEW["site_probabilities"].items():
        for site, p in zip("ABC", probs):
            assert site_given_group[(g, site)] / 1500 == pytest.approx(p, abs=0.04)
    for site in "ABC":
        for g in SKEW["site_probabilities"]:
            roles = Counter(r for u, (s, r) in a.items() if s == site and group_of[u] == g)
            n = sum(roles.values())
            assert roles["train"] / n == pytest.approx(0.7, abs=0.01)
            assert all(abs(roles[r] - 0.1 * n) <= 1 for r in ("calibration", "development", "test"))


def test_iid_keeps_roles_and_balances_sites():
    studies = synthetic()
    skew = partition.skew_partition(studies, SKEW)
    iid = partition.iid_partition(studies, skew, IID)
    assert all(iid[u][1] == skew[u][1] for u in skew)                       # same role in both partitions
    group_of = dict(studies)
    cell = Counter((s, r, group_of[u]) for u, (s, r) in iid.items())
    for r in partition.ROLES:
        for g in set(group_of.values()):
            sizes = [cell[(s, r, g)] for s in "ABC"]
            assert max(sizes) - min(sizes) <= 1


def test_bad_probability_table_is_rejected():
    bad = {**SKEW, "site_probabilities": {**SKEW["site_probabilities"], "normal": [0.5, 0.25, 0.2]}}
    with pytest.raises(ValueError):
        partition.skew_partition(synthetic(40), bad)


# ---------------------------------------------------------------- manifests and the whole pipeline

def write_raw(tmp_path, n=400):
    mesh = ["normal", "Cardiomegaly/mild", "Opacity/lung/base", "Spine/degenerative"]
    with open(tmp_path / "indiana_reports.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(REPORT_COLUMNS)
        for i in range(1, n + 1):
            findings = "The heart is normal in size. The lungs are clear." if i % 10 else ""
            if i % 10 == 1:
                findings = "XXXX XXXX. Heart XXXX."                         # too short once cleaned
            w.writerow([i, mesh[i % 4], mesh[i % 4].split("/")[0], "Chest", "", "", findings, "Normal."])
    with open(tmp_path / "indiana_projections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PROJECTION_COLUMNS)
        for i in range(1, n + 1):
            if i % 10 != 2:                                                 # every tenth study has no frontal
                w.writerow([i, f"{i}_IM-2.png", "Frontal"])
                w.writerow([i, f"{i}_IM-1.png", "Frontal"])
            w.writerow([i, f"{i}_IM-3.png", "Lateral"])
    return str(tmp_path)


def test_pipeline_exclusions_and_manifests(tmp_path):
    res = prepare.build(write_raw(tmp_path), GROUPS, SKEW, IID)
    assert res["n_total"] == 400
    assert res["excluded"] == {"empty_findings": 40, "too_short_after_cleaning": 40, "no_frontal_image": 40}
    assert len(res["rows"]) == 280
    assert res["rows"]["3"]["image"] == "3_IM-1.png"                        # first frontal by filename order

    out = str(tmp_path / "manifests")
    for name, assignment in (("skew_v1", res["skew"]), ("iid_v1", res["iid"])):
        manifest.write_partition(out, name, res["rows"], assignment, ["A", "B", "C"], res["meta"])
    skew = manifest.verify_partition(out, "skew_v1")
    iid = manifest.verify_partition(out, "iid_v1")
    assert set(skew) == set(iid) == set(res["rows"])                        # every eligible study, once
    assert all(skew[u][1] == iid[u][1] for u in skew)                       # no role differs across partitions

    assert len(manifest.load_split(out, "skew_v1", "A", "train")) > 0
    with pytest.raises(PermissionError):
        manifest.load_split(out, "skew_v1", "A", "test")
    with pytest.raises(PermissionError):
        manifest.load_split(out, "skew_v1", "A", "test", unlock_test=True)  # freeze tag still missing
    assert manifest.load_split(out, "skew_v1", "A", "test", unlock_test=True, freeze_tag="freeze-v1") is not None


def test_tampered_manifest_is_detected(tmp_path):
    res = prepare.build(write_raw(tmp_path), GROUPS, SKEW, IID)
    out = str(tmp_path / "manifests")
    manifest.write_partition(out, "skew_v1", res["rows"], res["skew"], ["A", "B", "C"], res["meta"])
    path = tmp_path / "manifests" / "skew_v1" / "B_train.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")                           # drop one study
    with pytest.raises(ValueError, match="recorded hash"):
        manifest.verify_partition(out, "skew_v1")

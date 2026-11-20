"""Phase 1 pipeline: raw CSVs -> eligible studies -> partition groups -> site partitions.

Eligibility is pre-registered (plan section 5.2). A study is excluded for the FIRST rule it fails:
  empty_findings              the FINDINGS field is empty
  too_short_after_cleaning    fewer than 5 words remain after clean_v1
  no_frontal_image            no image marked Frontal
  image_unreadable            the chosen image does not open (only checked when images are present)
The image used is the first Frontal image by filename order. Lateral images are unused in v1.
"""
import os
from collections import Counter

from clinguard.data import clean, ingest, labels, partition
from clinguard.utils.hashing import sha256_file, sha256_text

EXCLUSION_ORDER = ["empty_findings", "too_short_after_cleaning", "no_frontal_image", "image_unreadable"]


def _image_opens(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build(raw_dir, groups_cfg, skew_cfg, iid_cfg, images_dir=None, status="candidate"):
    """Returns a dict with the eligible rows, both site assignments and everything the data card reports."""
    studies = ingest.load_studies(raw_dir)
    rows, excluded, exclusion_uids = {}, Counter(), {}
    for s in studies:
        cleaned = clean.clean_findings(s.findings_raw)
        if not s.findings_raw.strip():
            reason = "empty_findings"
        elif clean.word_count(cleaned) < clean.MIN_WORDS:
            reason = "too_short_after_cleaning"
        elif not s.frontal:
            reason = "no_frontal_image"
        elif images_dir and not _image_opens(os.path.join(images_dir, s.frontal[0])):
            reason = "image_unreadable"
        else:
            rows[s.uid] = dict(image=s.frontal[0], group=labels.assign_group(s.mesh, groups_cfg),
                               needs_prior=clean.needs_prior(cleaned), findings=cleaned,
                               n_frontal=len(s.frontal), mesh=s.mesh, masked=clean.MASK in s.findings_raw)
            continue
        excluded[reason] += 1
        exclusion_uids[s.uid] = reason

    pairs = [(u, r["group"]) for u, r in rows.items()]
    skew = partition.skew_partition(pairs, skew_cfg)
    iid = partition.iid_partition(pairs, skew, iid_cfg)
    text_hash = sha256_text("\n".join(f"{u}\t{rows[u]['findings']}" for u in sorted(rows, key=int)))
    meta = dict(
        source="IU X-ray (Indiana University / Open-i), Kaggle copy raddar/chest-xrays-indiana-university, version 2",
        raw_sha256={f: sha256_file(os.path.join(raw_dir, f)) for f in (ingest.REPORTS_FILE, ingest.PROJECTIONS_FILE)},
        clean_version=clean.CLEAN_VERSION, prior_version=clean.PRIOR_VERSION, groups_version=groups_cfg["version"],
        split_seed=skew_cfg["split_seed"], cleaned_text_sha256=text_hash,
        image_check="passed" if images_dir else "not_run",
        status=status,               # "candidate" until gate G1, then "frozen"
    )
    return dict(n_total=len(studies), n_images=sum(s.n_images for s in studies), rows=rows,
                excluded=excluded, exclusion_uids=exclusion_uids, skew=skew, iid=iid, meta=meta)

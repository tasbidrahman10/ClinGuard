"""Read the IU X-ray CSV files (Kaggle copy `raddar/chest-xrays-indiana-university`).

One uid = one study. IU X-ray has no patient identifier, so a study is treated as the patient
unit (limitation recorded in docs/DATA_CARD.md).
"""
import csv
import os
from dataclasses import dataclass, field

REPORT_COLUMNS = ["uid", "MeSH", "Problems", "image", "indication", "comparison", "findings", "impression"]
PROJECTION_COLUMNS = ["uid", "filename", "projection"]
REPORTS_FILE = "indiana_reports.csv"
PROJECTIONS_FILE = "indiana_projections.csv"


@dataclass
class Study:
    uid: str
    mesh: str
    problems: str
    findings_raw: str
    frontal: list = field(default_factory=list)      # frontal image filenames, sorted
    n_images: int = 0


def _read(path, expected):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected:
            raise ValueError(f"{os.path.basename(path)}: expected columns {expected}, found {reader.fieldnames}")
        return list(reader)


def load_studies(raw_dir):
    """Returns all studies, sorted by numeric uid, with their frontal image filenames attached."""
    reports = _read(os.path.join(raw_dir, REPORTS_FILE), REPORT_COLUMNS)
    projections = _read(os.path.join(raw_dir, PROJECTIONS_FILE), PROJECTION_COLUMNS)
    studies = {}
    for r in reports:
        if r["uid"] in studies:
            raise ValueError(f"duplicate uid {r['uid']} in {REPORTS_FILE}")
        studies[r["uid"]] = Study(uid=r["uid"], mesh=r["MeSH"] or "", problems=r["Problems"] or "",
                                  findings_raw=r["findings"] or "")
    for p in projections:
        if p["uid"] not in studies:
            raise ValueError(f"image {p['filename']} belongs to unknown uid {p['uid']}")
        s = studies[p["uid"]]
        s.n_images += 1
        if p["projection"] == "Frontal":
            s.frontal.append(p["filename"])
    for s in studies.values():
        s.frontal.sort()
    return sorted(studies.values(), key=lambda s: int(s.uid))

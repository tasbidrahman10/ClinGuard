# Run log

One dated line per run or build: purpose, result, anomalies, resumptions, failures. Failed and
abandoned runs stay listed.

| Date | What | Purpose | Result | Notes |
|---|---|---|---|---|
| 2026-11-11 | `scripts/prepare_data.py --raw-dir data/raw/iu_xray` | First build of the split manifests from the real CSV files | 3,851 studies, 3,199 eligible; `skew_v1` hash `f0522cdb8c871902`, `iid_v1` hash `e3e300185b13507e` | Candidate manifests. Image-open check not run (images not downloaded). Raw SHA-256: reports `cd725771...`, projections `7af9c8a6...` |
| 2026-11-11 | `scripts/prepare_data.py --check` | Confirm the manifests are reproduced from the raw files | Both partitions reproduced exactly | |
| 2026-11-11 | `pytest -q` (Python 3.11.18, torch 2.15.0 CPU, Windows 11) | Engine and data tests | 49 passed | First run had 1 failure: the toy end-to-end test used an impossible pass mark (the toy sites disagree by design, so the loss cannot approach zero). The test now compares against the pooled least-squares optimum. No engine code changed. |
| 2026-11-20 | Laptop setup: `.venv` (CPython 3.11.18 via uv), `requirements-dev.lock`, `pytest -q` | Second development machine | 49 passed (torch 2.15.0+cpu, Windows 11) | Same versions as the 2026-11-11 desktop run |
| 2026-11-20 | Full Kaggle archive downloaded to `data/raw/iu_xray/` | Images for the image-open check | 7,470 PNGs + 2 CSVs; CSV SHA-256 identical to the 2026-11-11 build | |
| 2026-11-20 | `scripts/prepare_data.py --raw-dir data/raw/iu_xray --images-dir data/raw/iu_xray/images/images_normalized --check` | Image-open check (last eligibility rule) | All 3,199 chosen frontal images open; `skew_v1` and `iid_v1` reproduced exactly; 58 s | Manifests and data card not yet regenerated with `image_check: passed`; done at the G1 freeze |

No model training run has been made yet.

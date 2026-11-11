# Run log

One dated line per run or build: purpose, result, anomalies, resumptions, failures. Failed and
abandoned runs stay listed.

| Date | What | Purpose | Result | Notes |
|---|---|---|---|---|
| 2026-11-11 | `scripts/prepare_data.py --raw-dir data/raw/iu_xray` | First build of the split manifests from the real CSV files | 3,851 studies, 3,199 eligible; `skew_v1` hash `f0522cdb8c871902`, `iid_v1` hash `e3e300185b13507e` | Candidate manifests. Image-open check not run (images not downloaded). Raw SHA-256: reports `cd725771...`, projections `7af9c8a6...` |
| 2026-11-11 | `scripts/prepare_data.py --check` | Confirm the manifests are reproduced from the raw files | Both partitions reproduced exactly | |
| 2026-11-11 | `pytest -q` (Python 3.11.18, torch 2.15.0 CPU, Windows 11) | Engine and data tests | 49 passed | First run had 1 failure: the toy end-to-end test used an impossible pass mark (the toy sites disagree by design, so the loss cannot approach zero). The test now compares against the pooled least-squares optimum. No engine code changed. |

No model training run has been made yet.

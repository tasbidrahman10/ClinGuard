# ClinGuard-DiLoCo

Clinically constrained, low-communication training for multi-site chest X-ray report generation.
CSE499A/B Senior Capstone Design, North South University.
Team: Tasbid Al Rahman, Tazvir Rahat, Puja Ghosh. Supervisor: Dr. Sumaiya Tabassum Nimi.

This is research code for a study protocol, not a clinical tool. Nothing here establishes clinical
safety, diagnostic validity or privacy guarantees. "ClinGuard" is a working label.

## Where things are

| Path | What it is |
|---|---|
| `docs/IMPLEMENTATION_PLAN.md` | The plan: decisions, phases, gates, stop rules. Read section 0 first. |
| `docs/ClinGuard_DiLoCo_Research_Manual.pdf` | The study protocol the plan implements |
| `docs/DECISIONS.md` | Fixed decisions, verified assumptions, implementation choices |
| `docs/DEVIATIONS.md` | Every departure from the manual or the plan, dated |
| `docs/DATA_CARD.md` | Dataset counts, exclusions, partitions, limitations (generated) |
| `docs/RUNLOG.md` | One dated line per run or build |
| `docs/progress/` | Progress updates for the supervisor |
| `configs/data/` | Group lists and site-partition tables |
| `manifests/` | Split manifests: which study is in which site and data role |
| `src/clinguard/data/` | Ingest, cleaning, group labels, partitions, manifests, sampler |
| `src/clinguard/engine/` | DiLoCo round loop, outer step, weighting, checkpoints, cost accounting |
| `tests/` | Unit tests, including the manual's Appendix A tests |

## Status (2026-11-11)

| Phase | State |
|---|---|
| 0 Setup | Repository, tests and decision log in place. Open: private GitHub and Hugging Face repos, metric packages checked on Kaggle. |
| 1 Data | Pipeline built and run on the real CSV files; manifests are **candidates**. Open: image-open check on Kaggle, team review, model-side loader. |
| 2 Engine | Round loop, Nesterov outer step, uniform and fixed-q weighting, checkpoint and resume, byte accounting. Tests A.1, A.2, A.3, A.7 pass on toy problems. Open: random and loss-weighted baselines. |
| 3 to 8 | Not started. |

No gate has been declared passed yet.

## Setup and tests

```
conda create -p .venv python=3.11 pip
.venv/python -m pip install -r requirements-dev.lock
.venv/python -m pytest -q
```

The tests need no data and no GPU. They run from the repository root.

## Rebuilding the data manifests

The IU X-ray files are not in this repository (licence CC BY-NC-ND 4.0). Download
`indiana_reports.csv` and `indiana_projections.csv` from the Kaggle dataset
`raddar/chest-xrays-indiana-university` into `data/raw/iu_xray/`, then:

```
python scripts/prepare_data.py --raw-dir data/raw/iu_xray           # rebuild manifests and data card
python scripts/prepare_data.py --raw-dir data/raw/iu_xray --check   # confirm committed manifests are reproduced
```

## Rules that are easy to break

- The internal test split stays locked until the freeze. `load_split` refuses it without a freeze tag.
- Tune only on development data. Never pick a seed or checkpoint after seeing results.
- A fixed decision changes only through a dated entry in `docs/DEVIATIONS.md`.
- Results made with unreviewed probes are "engineering only" and are never reported as findings.

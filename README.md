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

## Status (2026-11-22)

| Phase | State |
|---|---|
| 0 Setup | Repository, tests and decision log in place. Open: private Hugging Face repo shared by the three Kaggle accounts, metric packages checked on Kaggle (G0). |
| 1 Data | **G1 passed (2026-11-20).** Manifests frozen: `skew_v1` `fba077ca06499e73`, `iid_v1` `77e5cd83bdf165f8`; all 3,199 chosen images open. |
| 2 Engine | **G2 met.** Round loop, Nesterov outer step, uniform / fixed-q / external / loss / random weighting, checkpoint and resume, byte accounting. Tests A.1, A.2, A.3, A.7 pass on toy problems. |
| 3 Model | **In progress.** Local half done: fixed-basis LoRA, examples and batches, the model-backed site, generation, test A.4; the full pipeline runs and resumes bit for bit on a tiny Qwen2.5-VL with real images. Kaggle sessions 1–2 done: 4-bit fp16 training is stable, 4.3 s per step in a real run, resume works (the 0.3% gap is GPU nondeterminism). **G3 not yet passed:** the site-A model collapsed to a normal template and only matches a model that always writes "normal" (I27). Next: Kaggle session 3 (`notebooks/kaggle_session3.ipynb`): learning rate, LoRA capacity, site B, and an image test. |
| 4 to 8 | Not started. |

Plan v1.1 (2026-11-20): no radiologist in CSE499A, so probes are automatically derived ("silver") and the
CSE499A endpoint is an automatic proxy; see `docs/IMPLEMENTATION_PLAN.md`.

## Setup and tests

Python 3.11 (matches CI). With [uv](https://docs.astral.sh/uv/):

```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.lock     # .venv/bin/python on Linux/macOS
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe -m pytest -q
```

Or with conda: `conda create -p .venv python=3.11 pip`, then `pip install -r requirements-dev.lock`.

The model tests (tiny Qwen2.5-VL on CPU, real images) also need `requirements-model.lock`; without it,
or without the IU X-ray files in `data/raw/iu_xray/`, those tests skip.

Scoring generated reports (CheXbert, RadGraph, the worst-site proxy) runs in a separate CPU environment,
because the metric packages need transformers 4.x: see `requirements-eval.lock` and `scripts/evaluate.py`.

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
- Results made with silver probes are always labelled "automatically derived probes, not expert-verified".
- The repository is public: never commit checkpoints, generated reports, cleaned text or probe sentences.

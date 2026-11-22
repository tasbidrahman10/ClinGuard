# ClinGuard-DiLoCo — Implementation Plan (v1.1)

Clinically constrained, low-communication training for multi-site chest X-ray report generation.
Capstone CSE499A/B, North South University. Team: Tasbid Al Rahman, Tazvir Rahat, Puja Ghosh. Supervisor: Dr. Sumaiya Tabassum Nimi.
Plan date: 2026-11-10, **revised 2026-11-20 (v1.1)**. Source protocol: "ClinGuard-DiLoCo Methods Manual 01, v1.0" (the **manual**). Section numbers like "manual §6.4" refer to it.

**What changed in v1.1 (2026-11-20).**
1. **No radiologist in CSE499A** (supervisor). Probes are built automatically and accepted only when two independent label sources agree ("silver" probes, D15). The CSE499A clinical endpoint is the automatic worst-site proxy (D16). A radiologist audit of a probe sample and the blinded reader pilot move to CSE499B. Logged as deviations M6 and M7.
2. **Compute is shared across the team's three Kaggle accounts**: whole runs are distributed between accounts, and all checkpoints go to one private Hugging Face repo (Section 4.1).
3. **The GitHub repository is public** (D17). It never holds data-derived text or checkpoints.
4. **Roles**: Tasbid leads all development; the other two members operate Kaggle runs (Section 13).
5. **Semester split** made explicit (Section 1, Section 12). The internal test split stays locked through CSE499A.

**Revision 2026-11-22 (G3).** The G3 learning criterion now compares against a model that always writes a normal report, not only against zero-shot (I27, `docs/DECISIONS.md`); the LoRA target set became a run setting so a wider set can be tested (I28).

**This plan supersedes the earlier PoolTrain / BatchNorm direction** (old `CLAUDE.md`). Only the engineering lessons carry over: a tested Nesterov outer step, per-worker optimizer state, exact checkpoint/resume, pre-registered criteria, and honest reporting.

---

## 0. How to use this document (read first — coding agent and humans)

1. Work **phase by phase** (Section 12). Do not start a phase until the previous phase's **gate** passes. Gates exist so we never have to look back.
2. Every item marked **[DECISION Dn]** is fixed. Do not change it silently. If it must change, add a dated entry to `docs/DEVIATIONS.md` (what, why, who approved).
3. Every item marked **[ASSUMPTION]** or **[VERIFY]** is unconfirmed. Verify it at the stated gate and record the result in `docs/DECISIONS.md`.
4. Every item marked **[ASK PROF]** needs the supervisor's answer (Section 2.3). A default is given so work is never blocked.
5. Never touch the **internal test split** before the freeze (Phase 7). Code must refuse to load it unless `--unlock-test` is passed together with a freeze tag.
6. Never tune on, build probes from, or select checkpoints with test data. Never pick the best seed after the fact.
7. Numbers in this plan that are estimates (compute time, dataset counts, parameter counts) are labelled as estimates. Replace them with measured values as soon as they exist.
8. Report negative results. The manual's stop rules (Section 15) are binding.
9. The GitHub repository is **public** (D17). Never commit anything derived from the data: checkpoints, generated reports, cleaned report text, probe sentences, per-case outputs. Never put a token in code or in a notebook; tokens live only in Kaggle Secrets or local environment variables.

---

## 1. The project in one page

**Goal.** Fine-tune one vision-language model (VLM) to write the FINDINGS section of chest X-ray reports, across several simulated hospitals that keep their data local, using DiLoCo (many local AdamW steps, rare synchronization, Nesterov outer step). Compare ordinary DiLoCo with **ClinGuard**: an outer aggregation rule that re-weights hospital updates using small, radiologist-reviewed "probe" sets at each hospital, so that no hospital's clinically important errors get worse because of another hospital's update.

**Primary question (manual §1).** At a fixed data, compute and communication budget, does clinical-aware weighting reduce the **worst site's** clinically significant report-error rate relative to tuned fixed-weight DiLoCo, while keeping overall report quality acceptable?

**Research questions (manual §4).**
- RQ1 — Does the problem exist? Do per-site clinical errors change as H (local steps) grows?
- RQ2 — Can it be predicted? Do sketch-based direction scores predict actual probe-loss changes?
- RQ3 — Does the intervention help, against uniform, fixed, loss-weighted, random and probe-data-only baselines?
- RQ4 — At what cost (GPU hours, memory, bytes, rounds)?

**What we are building.** A single-machine **simulation** of K hospitals plus a coordinator (no real network in v1), with exact byte accounting, run on Kaggle T4 GPUs.

**Scope v1 (manual §1).** Synchronous rounds, full participation, one shared architecture and checkpoint, English chest X-rays, one report section, one shared global model. Out of scope: asynchrony, personalization, real networking, heterogeneous architectures, formal privacy guarantees.

**Semester split.**
- **CSE499A (this semester):** everything that needs no radiologist. Data, engine, model integration, baselines and the RQ1 sweep (gates G0–G4, required). Then silver probes and ClinGuard on development data with automatic metrics (G5–G6, if time allows). All results are on **development data** and carry the labels "automatic metrics" and "automatically derived probes".
- **CSE499B:** a radiologist audit of a sample of the silver probes (Section 7.3), any reruns the audit makes necessary, the analysis freeze, the **one** internal-test unlock, and the blinded reader pilot if readers are available (Section 10.4).
- The test split stays locked until after the probe audit. If the audit changes the probes, the method changes, and a test set already opened would be spent.

---

## 2. Decisions, assumptions and open questions

### 2.1 Fixed decisions (with reasons)

| ID | Decision | Why | Alternative rejected |
|---|---|---|---|
| D1 | **Dataset: IU X-ray (Indiana University / Open-i)** from the public Kaggle copy `raddar/chest-xrays-indiana-university` | Supervisor's instruction; public; small enough for free GPUs | MIMIC-CXR / CheXpert Plus (credentialing, compute) |
| D2 | **Sites are simulated**: K = 3 silos made by a deliberately skewed partition of IU X-ray (Section 5.4), plus an IID partition as a control | One source only; skew is needed for site trade-offs to exist at all; manual §9.3 "dataset shift" | K = 2 (worst-site max is weak), K = 4 (too few studies per site) |
| D3 | **Task: generate FINDINGS from one frontal image**, fixed prompt | Manual §5.1 frontal-image selection; halves visual tokens versus two views | Findings + impression; two-view input |
| D4 | **Base model: a general-purpose ~3B VLM, primary candidate `Qwen/Qwen2.5-VL-3B-Instruct`**, 4-bit (QLoRA-style) | Fits a 16 GB T4; team has Qwen QLoRA experience; general models are less likely than radiology models to have been trained on IU X-ray | 7B (about 2x compute); MedGemma / CheXagent / MAIRA-2 as primary (radiology pretraining, contamination risk) |
| D5 | **Trainable parameters φ = LoRA B matrices only; LoRA A is frozen and identical at all sites** (fixed-basis LoRA, rank 16, on the language model's attention projections) | Manual §5.3: with common A, Σ wᵢBᵢA = (Σ wᵢBᵢ)A, so weighted averaging is exact; φ is small (a few million numbers) so exact dot products are cheap | Trainable A and B (averaging mismatch; kept as ablation) |
| D6 | **Visual connector frozen in the primary configuration**; connector training is an ablation | Keeps φ purely linear-in-B and the 4-bit setup simple. **Deviation from manual §9.2** (which lists connector parameters); logged | Train connector from the start |
| D7 | **Engine conventions**: dᵢ = φ_t − φᵢ^(H); m_t = μ m_{t−1} + d̄_t; φ_{t+1} = φ_t − η_out (d̄_t + μ m_t); m_{−1} = 0; per-site AdamW state persists across rounds | Manual Eq. (1)–(2); DiLoCo | PyTorch SGD(nesterov) used only as a cross-check in tests |
| D8 | **Round structure**: H = 100 optimizer steps (primary), sensitivity H ∈ {25, 250}; total local optimizer steps per site fixed at 1,000 (so T = 10, 40, 4 rounds) | Manual §9.2; fixed-data comparison for RQ1 | Fixed number of rounds |
| D9 | **Exact scores are the reference; sketches are the communicated approximation.** Both are always computed and logged in simulation | φ is small, so exact ⟨g, d⟩ is free here; this answers RQ2 directly (manual §6.3) | Sketch only |
| D10 | **Probe groups in the pilot: one site-wide group per site** (J = K = 3 constraints) | Manual §9.2; IU is too small for per-finding groups | Per-finding groups (later, if counts allow) |
| D11 | **Constraint bound** b_{j} = ℓ_j(φ_t) + δ, evaluated on the same probe batch (predicted probe loss may not rise more than δ above its current value) | A fixed early reference becomes permissive as the model improves (manual §6.4 warns of this) | Fixed reference checkpoint — kept as config option `bound_mode: fixed_ref` **[ASK PROF Q5]** |
| D12 | **Seeds**: 11 (primary, pre-selected), 22, 33 (replications). Data split seed 20261110, never changed | Manual §9.1 | Choosing seeds after results |
| D13 | **Compute: Kaggle T4 on the team's three Kaggle accounts, distributed by whole run (Section 4.1); checkpoints and logs in one private Hugging Face repo**, full state resume | Free; proven by the team on EFSM; three accounts triple the weekly GPU quota | Paid cloud; splitting one run's rounds across accounts (rounds are sequential, so no speed-up) |
| D14 | **Code: one Python package, config-driven (YAML), pure PyTorch + transformers + peft + bitsandbytes**; no federated framework | Full control of optimizer states and the outer step; fewer hidden behaviours | Flower / FedML |
| D15 | **Probes in CSE499A are silver probes.** They are built automatically from the calibration split and accepted only when the human-coded MeSH field and CheXbert labels of the radiologist-written report agree (Section 7.2). A radiologist audits a random sample in CSE499B (Section 7.3) | Supervisor, 2026-11-20: no radiologist this semester. Using two independent sources and requiring agreement filters most label noise | Waiting for a radiologist (blocks Phases 5–6); single-source labels |
| D16 | **CSE499A clinical endpoint is automatic**: worst-site proxy error rate (Section 10.3) with CheXbert-label F1 and RadGraph F1. The blinded reader pilot (Section 10.4) moves to CSE499B | Supervisor, 2026-11-20 | Reader study this semester |
| D17 | **The GitHub repository is public.** It holds code, configs, documentation and the uid-only manifests. Checkpoints, generated reports, cleaned text and probe sentences go to the private HF repo or stay local | Kaggle notebooks can clone without a token. Licence A2 forbids redistributing data-derived text | Private GitHub repo (plan v1.0) |

### 2.2 Assumptions to verify (each has a gate)

| ID | Assumption | Verify at |
|---|---|---|
| A1 | The Kaggle IU copy has `indiana_reports.csv` (uid, MeSH, Problems, image, indication, comparison, findings, impression), `indiana_projections.csv` (uid, filename, projection) and PNG images; about 3,850 studies / 7,470 images | G1 |
| A2 | IU X-ray licence is CC BY-NC-ND 4.0 (as stated on public mirrors). Consequence: research use only, **do not publish modified copies of the data; keep trained models and outputs private unless the supervisor approves** | G0 |
| A3 | `Qwen2.5-VL-3B-Instruct` licence permits research use (believed to be a research-only licence; the 7B is believed Apache-2.0). Record the exact licence and model revision hash | G0 |
| A4 | The base model was not trained on IU X-ray. This **cannot be proven** for a web-trained general model; record as a limitation and measure the zero-shot baseline | G3 |
| A5 | The model trains stably in 4-bit with fp16 compute on a T4 (T4 has no bf16). Fallbacks, in order: fp32 LoRA compute; `Qwen/Qwen2-VL-2B-Instruct`; another ~2–4B general VLM | G3 |
| A6 | One full run (1,000 steps × 3 sites, effective batch 4, plus evaluation) takes roughly 4–6 T4-hours. **Estimate only**; the experiment grid in Section 11 is sized from the measured value | G3 |
| A7 | `radgraph` and `f1chexbert` (or equivalents) install and run on Kaggle | G0 |
| A8 | Eligible studies (non-empty findings + a frontal image) number roughly 3,300. **Estimate only** | G1 |
| A9 | Each of the three team Kaggle accounts is phone-verified and has the weekly GPU quota (about 30 hours per account). Each account can clone the repo and write to the shared HF repo | G0 |
| A10 | All accounts run the same environment (Kaggle image pinned, `requirements.lock`). A run resumed on a different machine continues correctly. On GPU it is **close to, but not bit-identical with**, an uninterrupted run, because fp16 kernels are not deterministic. Measure the gap with a handoff test | G3 |
| A11 | CheXbert (`f1chexbert` or equivalent) labels the ~330 calibration reports reliably enough to serve as the second label source for silver probes. Check against MeSH agreement rates | G5 |

### 2.3 Questions for the supervisor (defaults in bold so work is not blocked)

- Q1. External test set: IU X-ray is now the training source, so the manual's external set is gone. **Default: no external site; stated as a limitation.** Alternative: a second public dataset later.
- Q2. Radiologist time. **Answered 2026-11-20: no radiologist in CSE499A.** Use automatic labels and automatic metrics. In CSE499B a radiologist may audit a sample of probes and run the reader pilot (D15, D16; deviations M6, M7).
- Q3. Baseline depth: reproducing FedMRG / FedIHRAS / MERIT / IsoLoCo is heavy. **Default: Tier 1 and Tier 2 baselines (Section 9) are required; prior-art reproductions are Tier 3 / stretch.**
- Q4. Final method name ("ClinGuard" collides with an existing paper, manual §3). **Default: keep as working label.**
- Q5. Constraint bound definition (D11). **Default: relative-to-current.**
- Q6. Ethics / IRB for the reader study with public de-identified data. **Deferred to CSE499B**, together with the reader pilot.
- Q7. Which semester boundary applies (what must exist at the end of CSE499A). **Default: gates G0–G4 required; G5–G6 on silver probes if time allows; test unlock and reader pilot in CSE499B.**
- Q8. Size of the CSE499B radiologist probe audit. **Default: a stratified random sample of 20 accepted silver probes per site (60 in total).**

---

## 3. Architecture

```
            ┌───────────────────────── one process, one GPU ─────────────────────────┐
            │  Base VLM (4-bit, frozen) + LoRA A (frozen, shared)   loaded ONCE      │
            │                                                                        │
 round t:   │  for each site i (sequential):                                         │
            │     load φ_t into LoRA B                                               │
            │     PROBE:  loss ℓ_i(φ_t), gradient g_i   (reviewed calibration set)   │
            │     restore site i's AdamW + sampler state                             │
            │     TRAIN:  H optimizer steps on site i's training split               │
            │     d_i = φ_t − φ_i^(H);  save site i's AdamW + sampler state          │
            │     "send": sketch(d_i), sketch(g_i), loss, counts   (bytes counted)   │
            │  COORDINATOR:                                                          │
            │     scores  M[j,i] = <g_j, d_i>   (exact and sketched)                 │
            │     weights w = solve_controller(M, momentum terms, q)                 │
            │     d̄ = Σ w_i d_i ;  m ← μ m + d̄ ;  φ ← φ − η_out (d̄ + μ m)           │
            │     checkpoint + log                                                   │
            └────────────────────────────────────────────────────────────────────────┘
```

Baselines are the same loop with a different `weighting` (uniform, fixed-q, random, loss-weighted) or a different `outer` (FedAvg), so comparisons are controlled by construction.

### 3.1 Repository layout

```
clinguard/
  README.md  pyproject.toml  requirements.lock
  configs/            base.yaml  model/  data/  method/  experiment/ (incl. grid.yaml, the job list)
  docs/               DECISIONS.md  DEVIATIONS.md  RUNLOG.md  DATA_CARD.md  CODEBOOK.md
  src/clinguard/
    data/             ingest.py  clean.py  labels.py  partition.py  manifest.py  dataset.py  collate.py
    model/            load.py  lora.py  params.py  generate.py
    engine/           inner.py  outer.py  weighting.py  state.py  accounting.py  runner.py
    probe/            candidates.py  review_io.py  loss.py  gradient.py
    controller/       sketch.py  scores.py  solver.py
    eval/             metrics.py  proxy.py  reader_package.py  stats.py
    utils/            seed.py  hashing.py  logging.py  hf_sync.py
  scripts/            prepare_data.py  profile_model.py  run.py  run_job.py  evaluate.py
                      make_silver_probes.py  make_review_pack.py  make_reader_pack.py
  tests/              test_sign.py  test_momentum.py  test_reduction.py  test_lora_identity.py
                      test_sketch.py  test_constraint.py  test_resume.py  test_manifest.py  test_toy_e2e.py
  notebooks/          kaggle_runner.ipynb
```

Rule: `engine/` and `controller/` never import `transformers`. They operate on a flat parameter vector and a `SiteTask` interface (`probe()`, `train(H)`, `get/set_params`). This lets every Appendix-A test run on a toy model on CPU in seconds.

---

## 4. Environment and workflow

- Python 3.11; pin exact versions of torch, transformers, peft, bitsandbytes, accelerate in `requirements.lock`. Record the Kaggle image, GPU model and CUDA version in every run log.
- Code lives in the **public** GitHub repo `tasbidrahman10/ClinGuard` (D17). The Kaggle notebook clones a specific commit hash; the hash is written into the run log. No code edits inside Kaggle. Kaggle notebooks themselves stay **private** on Kaggle, since their outputs can contain generated reports.
- Checkpoints and logs go to **one private** Hugging Face repo, path `runs/<run_id>/`. `run_id = <method>_<partition>_H<H>_s<seed>_<cfg-hash8>`.
- A run is resumable at round granularity: rerunning the same command continues from the last complete round and must reproduce an uninterrupted run (test A.7; on GPU see A10).
- Kaggle provides two T4s: run two independent runs in parallel (one per GPU), never one run across both.
- Three test tiers: (a) toy quadratic / tiny MLP on CPU (unit tests, CI on every commit); (b) a tiny model with the **same Qwen2.5-VL architecture** (`trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration`, I17) on real IU images, for pipeline tests on CPU; (c) the real model.
- Local machines (Tasbid's desktop and laptop) have no usable GPU for the 3B model; they run tiers (a) and (b), the data pipeline, probe construction and analysis. Every tier (c) run happens on Kaggle.

### 4.1 Shared compute across three Kaggle accounts (D13)

The team's three Kaggle accounts run jobs at the same time, and all of them write to the one private HF repo.

- **Parallel by run, not by round.** Round t+1 starts from the outer step of round t, so the rounds of one run are strictly sequential. Three accounts cannot train different rounds of the same run at once. The speed-up comes from running *different* runs at once (methods, seeds, H values, partitions). That is up to 3 accounts × 2 T4s = 6 runs in parallel. One run never uses two GPUs.
- **Handoff.** A run can still move between accounts when a session hits the 12-hour limit or an account's weekly quota runs out. The next operator resumes from the last complete round stored in the HF repo. A handoff keeps a run going; it does not make it faster.
- **Job list.** `configs/experiment/grid.yaml` (in GitHub) lists every planned run: run_id, config file, pinned code commit and assigned account. Tasbid edits it. Operators only start runs listed for their account.
- **HF layout.** `runs/<run_id>/` holds:
  - `checkpoint.pt`: latest complete round only; earlier rounds stay in the HF commit history.
  - `run.json` and `rounds.jsonl`.
  - `phi_final.safetensors`: written after the last round.
  - `LEASE.json`: who is training the run right now.
- **Lease.** Before training, `run_job.py` writes `LEASE.json`: Kaggle account, session id, code commit, start time, and an expiry 13 hours later. It refuses to start while another account holds an unexpired lease. The lease is renewed at every checkpoint upload and deleted when the run finishes. An expired lease may be taken over, and the takeover is logged.
- **Upload policy (I24).** After a round, the job uploads the run folder in **one commit together with the lease renewal**, at most every 15 minutes, and always at the final round and at a deliberate stop. A killed session loses at most about 15 minutes of compute, and short rounds (H = 25) do not flood the shared repo with commits.
- **Storage.** Each upload of `checkpoint.pt` (~85 MB) stays in the repo history. Once a run is finished only `phi_final`, `rounds.jsonl`, `run.json` and the generations are needed; old checkpoints are removed and the history squashed in a maintenance step taken when no lease is active.
- **Same everything.** The notebook clones the commit pinned in `grid.yaml`, installs `requirements.lock`, and records the Kaggle image, GPU, CUDA and package versions in `run.json`. Resuming is refused when the config hash, manifest hash or code commit differs from the checkpoint (I9, extended with the commit).
- **Access.**
  - Preferred: a free HF organization owns the private repo, and each member writes with their own token.
  - Fallback: one fine-grained token with write access to that single repo only.
  - Either way, tokens live only in Kaggle Secrets (`HF_TOKEN`) and can be revoked per member.
- **Determinism.**
  - On CPU, a resumed run is bit-identical to an uninterrupted one (A.7).
  - On GPU, a resumed or handed-off run has the same state and configuration but is not guaranteed bit-identical (A10).
  - Every handoff is listed in `RUNLOG.md`.
  - Gate G3 measures the size of the difference with a two-account handoff test.
- **Size.** φ has exactly 2,654,208 numbers (I18): rank-16 B matrices on q/k/v/o of Qwen2.5-VL-3B's 36 language-model layers. A checkpoint with φ, m and three sites' AdamW moments in fp32 is about 85 MB.

---

## 5. Data pipeline

### 5.1 Ingest (A1 to verify)
Read `indiana_reports.csv` and `indiana_projections.csv`; join on `uid`. One `uid` = one study. IU has no patient identifier, so **a study is treated as the patient unit**; state this limitation in `DATA_CARD.md` (manual §5.1 patient-level rule cannot be fully enforced). Adult-only eligibility also cannot be verified (no age field); state it.

### 5.2 Eligibility (pre-registered; log every exclusion with reason and counts)
Include a study if: (1) `findings` is non-empty and has at least 5 words after cleaning; (2) it has at least one image marked Frontal; (3) the image file opens. Image choice: the first Frontal by filename order. Lateral images are unused in v1.

### 5.3 Text cleaning (versioned script, `clean_v1`)
- Collapse whitespace; keep original casing and punctuation.
- IU de-identification writes `XXXX`. Remove sentences in which more than half the tokens are `XXXX`; otherwise delete the `XXXX` token and tidy spacing.
- Add a boolean `needs_prior` when the findings match a fixed regex list (`compared`, `prior`, `previous`, `unchanged`, `stable`, `interval`, `again`). Such studies stay in training and testing (references are not edited) but their sentences are **not** used as probe statements, and results are also reported excluding them (manual §5.1).
- Never write cleaned text back over raw data. Do not redistribute the cleaned data (licence A2).

### 5.4 Labels and site partition (D2)
Derive one **partition group** per study from the human-coded `MeSH` / `Problems` fields (not from the report text), by first match in this order:
1. `normal` — MeSH is "normal".
2. `cardio_pleural` — cardiomegaly, cardiac shadow enlarged, pleural effusion, pulmonary edema / congestion, aorta tortuous / atherosclerosis.
3. `parenchymal` — opacity, atelectasis, nodule, granuloma, infiltrate, consolidation, pneumonia, emphysema, cicatrix / scarring, mass.
4. `other_abnormal` — everything else (spine degeneration, fractures, devices, and so on).

The exact term lists live in `configs/data/groups_v1.yaml` and are frozen at G1.

**Skewed partition `skew_v1` (primary).** Each study is assigned to a site by sampling from this table (seed 20261110):

| Group | Site A ("screening") | Site B ("cardiac") | Site C ("pulmonary") |
|---|---|---|---|
| normal | 0.50 | 0.25 | 0.25 |
| cardio_pleural | 0.15 | 0.65 | 0.20 |
| parenchymal | 0.15 | 0.20 | 0.65 |
| other_abnormal | 0.34 | 0.33 | 0.33 |

**IID partition `iid_v1` (control).** Uniform random site assignment, same seed.
Every table and figure must say "simulated sites from one source" (manual §5.1).

### 5.5 Five data roles (manual §5.2)
Within each site, stratified by partition group: **70% train / 10% calibration / 10% development / 10% internal test**. No external test (Q1).
- Train: inner-loop fine-tuning only.
- Calibration: probe construction and the controller only.
- Development: all tuning and checkpoint/hyperparameter selection.
- Internal test: one locked evaluation after the freeze.

### 5.6 Manifests
`prepare_data.py` writes `manifests/<partition>/<site>_<role>.csv` (uid, image filename, group, needs_prior) plus `manifest.json` with SHA-256 of every file, the cleaning version, group-list version, seeds, and counts per site × role × group. `test_manifest.py` asserts: no uid in two roles or two sites; counts match; hashes match. The manifest hash is written into every run log.

### 5.7 Loader
Fixed prompt (frozen at G3, `prompt_v1`): system: "You are an expert chest radiologist." user: `<image>` + "Write the FINDINGS section of the radiology report for this frontal chest X-ray." Target: cleaned findings. Loss on target tokens only. Image resolution fixed (Section 6.1). Per-site sampler: epoch-wise shuffled permutation without replacement, with its own generator; its state is part of the checkpoint. Log unique studies consumed per round (manual §9.2: small sites repeat data at large H).

---

## 6. Model and trainable parameters

### 6.1 Loading (A3, A5)
- `Qwen/Qwen2.5-VL-3B-Instruct` at a **pinned revision hash**; processor/tokenizer versions recorded.
- 4-bit NF4 quantization (bitsandbytes), double quantization on, compute dtype fp16; gradient checkpointing on.
- Image tokens fixed: set `min_pixels = max_pixels = 448×448` (about 256 visual tokens). Profile 336 and 448 at G3 and freeze one.
- Generation settings (frozen at G3): greedy decoding, `max_new_tokens = 160`, repetition penalty 1.0.

### 6.2 Fixed-basis LoRA (D5)
- Targets: `q_proj, k_proj, v_proj, o_proj` in the **language model** layers only (assert that no vision-tower module is matched). Rank 16, alpha 32, dropout 0.
- Initialise with a dedicated seed (`lora_seed = 2026`), **the same for every run and site**: A random (PEFT default), B zero. Then set `requires_grad=False` on all A. Save A once (`lora_A.safetensors`, with checksum); every run loads this file and verifies the checksum.
- φ = all B matrices, flattened in **sorted parameter-name order** by `model/params.py` (`flatten`, `unflatten`, `get_phi`, `set_phi`). Print and log P = number of entries (estimate: a few million), tensor names and shapes.
- Global φ and momentum m are kept in fp32 on CPU; site copies are loaded to GPU for training.

### 6.3 Inner optimizer
AdamW (β = 0.9, 0.999; weight decay 0), gradient clipping 1.0, effective batch 4 (microbatch × accumulation from profiling), constant learning rate after a 50-step linear warm-up in round 0 only. **Learning-rate grid {5e-5, 1e-4, 2e-4}** on development data. Note: the manual's grid is {1e-5, 5e-5, 1e-4}; B-only LoRA usually needs a higher rate, so the grid is shifted — logged as a deviation, to be confirmed by the tuning result. An "optimizer step" is one AdamW update (not a microbatch).

---

## 7. Clinical probes (calibration set)

### 7.1 What a probe is
A triple (image x, correct statement y⁺, clinically contradictory statement y⁻), under the same prompt context c as generation. Example: y⁺ "There is a small left pleural effusion." / y⁻ "There is no pleural effusion."

### 7.2 Silver probe construction (`probe/candidates.py`, `scripts/make_silver_probes.py`; calibration split only; D15)
No radiologist reviews probes in CSE499A. Probes are built automatically. A probe is accepted only when **two independent label sources agree**:
- **Source 1:** the human-coded MeSH field. It was assigned by the dataset's annotators, not by our code or the report parser.
- **Source 2:** CheXbert labels (positive / negative / uncertain / blank) extracted from the **radiologist-written** FINDINGS text. The labeler version is pinned (A7, A11).

Steps:
1. **Template table** (`probe_templates_v1.yaml`, versioned): about 10 findings (pleural effusion, cardiomegaly, pneumothorax, focal opacity/consolidation, edema, atelectasis, nodule, normal heart size, clear lungs, and so on). Each finding has a matched pair of statements: a positive form and its negation. Sided findings also get left/right forms.
2. **Positive pair (finding present).** Accept only if MeSH codes the finding **and** CheXbert labels it positive. y⁺ is the template positive and y⁻ is the template negation. If MeSH gives a side (for example "Pleural Effusion/left") and the report text names the same side, also build a wrong-side pair.
3. **Negative pair (finding absent).** Accept only if CheXbert labels the finding **negative** (the radiologist explicitly ruled it out) **and** MeSH does not code it. y⁺ is the template negation and y⁻ is the template positive. **"Not mentioned" never counts as absent.**
4. **Exclude**, with a logged reason:
   - studies flagged `needs_prior`;
   - studies whose MeSH field is "No Indexing" (never coded by the annotators, so MeSH cannot serve as a source; G1 decision);
   - findings whose source sentence was changed by `clean_v1` (deleting `XXXX` can strengthen a statement);
   - any finding where the sources disagree or CheXbert says uncertain.
5. **Same wording style on both sides.** y⁺ and y⁻ come from the same template family and differ only in the clinical claim. The probe therefore cannot be won by preferring radiologist phrasing over template phrasing. A paraphrase set, with the report's own sentence as y⁺, is a secondary check (Section 7.4).
6. **Sampling.** Cover sites, normal/abnormal studies and finding types; record denominators. Cap each study at 3 pairs so no single study dominates.
7. **Freeze** as `probes_silver_v1.jsonl`, with a checksum and an exclusion table. It is stored locally and in the private HF repo, **never in GitHub** (D17).
8. **Labelling of results.** Results built on these probes are reported as findings of CSE499A. Each one carries the label "**automatically derived probes, not expert-verified**". Never describe them as reviewed or as clinical ground truth (manual §6.2).
9. **Eligibility gate per site:** **at least 50 accepted silver probe pairs**. Otherwise the site's group is "unsupported" and omitted from the controller; it is not given zero loss.

### 7.3 Radiologist audit (CSE499B; `make_review_pack.py`, `review_io.py`)
- If a radiologist is available, export a stratified random sample of accepted silver probes (Q8: default 20 per site). Each pack is self-contained: image, context, y⁺ and y⁻. Its fields are `y_plus_supported` (yes/no/insufficient), `y_minus_contradictory` (yes/no/insufficient) and `notes`. No model outputs are shown.
- Import the completed sheet. Report the agreement rate per site and per finding with exact denominators, and log disagreements.
- **The audit comes before the freeze and the test unlock.** If agreement is poor, build `probes_v2` from reviewed or corrected cases only, rerun the affected Phase 6 runs on development data, and then freeze.
- If no radiologist is available in CSE499B either, the final report says the probes were never expert-verified.

### 7.4 Probe loss and gradient (`probe/loss.py`, `gradient.py`; manual Eq. 3–4)
- s_φ(y | x, c) = mean token log-probability of y, teacher-forced.
- ℓ_g(φ) = mean over cases of [ −a·s(y⁺) + b·softplus((s(y⁻) − s(y⁺)) / τ) ], with a = b = 1, τ = 1 as starting values.
- Each round, each site draws a **probe batch of 24 accepted cases** (round-specific seed, stratified), computes ℓ_j(φ_t) and g_j = ∇_φ ℓ_j(φ_t) in eval mode, before local training. Probe compute (forward/backward counts and seconds) is logged separately for RQ4.
- Controls required before trusting the probe (manual §8.6): text-only (blank image) and image-swapped versions must **not** achieve the same probe loss; paraphrase check on a subset. With no radiologist in CSE499A, these controls are the main evidence that the silver probes measure something image-grounded, so they are part of gate G5.

---

## 8. Controller

### 8.1 Sketch (`controller/sketch.py`; manual §6.3)
CountSketch R_t: ℝ^P → ℝ^r with a bucket index h(p) ∈ [0, r) and sign s(p) ∈ {±1} generated from `sketch_seed(run_seed, round)`; the seed is drawn **after** the previous round is fixed and is independent of training/probe randomness. Default r = 1024; study r ∈ {256, 1024, 4096} and two hash seeds. Implement with `index_add_`; never build an r × P matrix. Sites "send" R d_i and R g_j; the coordinator sketches its own momentum.

### 8.2 Scores (`controller/scores.py`)
For every (probe group j, site i): exact M[j,i] = ⟨g_j, d_i⟩ and sketched M̂[j,i] = ⟨R g_j, R d_i⟩; also ⟨g_j, m_{t−1}⟩ exact and sketched. Log both, their absolute error, sign agreement and rank correlation each round (RQ2). `score_source: sketch | exact` selects which one drives the controller (primary: sketch; reference: exact).

### 8.3 Weight selection (`controller/solver.py`; manual Eq. 6–10)
- Actual Nesterov direction for candidate weights w: v_t(w) = μ² m_{t−1} + (1 + μ) Σ_i w_i d_i.
- Predicted probe loss: ℓ̂⁺_j(w) = ℓ_j(φ_t) − η_out [ μ² ⟨g_j, m_{t−1}⟩ + (1 + μ) Σ_i w_i M[j,i] ]  (linear in w).
- Solve: minimise KL(w ‖ q) + λ Σ_j ξ_j²  subject to ℓ̂⁺_j(w) ≤ b_j + ξ_j, ξ_j ≥ 0, Σ w_i = 1, (1 − ρ) q_i ≤ w_i ≤ (1 + ρ) q_i.
- q = fixed baseline weights from eligible training-study counts. b_j per D11.
- Implementation: `cvxpy` (exponential-cone solver such as Clarabel); fallback `scipy.optimize` SLSQP; **if both fail, use w = q and log `solver_status = fallback`**. Log per round: w, ξ, constraint residuals, status, solve time.
- Tuning grid on development data only: ρ ∈ {0.25, 0.5}, λ ∈ {1, 10, 100}, δ ∈ {0, 0.01}.
- Never describe ξ = 0 as a safety certificate (manual §6.4).

---

## 9. Methods to implement

All share model, data, trainable parameters, inner optimizer, H and tuning budget unless stated.

**Tier 1 — required**
| Method | Definition |
|---|---|
| Zero-shot | Base model, no training (contamination and floor reference) |
| Local-only | Each site trains alone for the same number of steps; no sync |
| FedAvg | Outer step φ ← φ − d̄ (η_out = 1, μ = 0), weights q; AdamW state reset each round (variant with persistent state reported separately) |
| Uniform DiLoCo | w_i = 1/K, Nesterov outer step |
| **Fixed-q DiLoCo** | w = q, Nesterov outer step — **primary comparator** |
| **ClinGuard** | Section 8 weights, same backbone — proposed method |
| Probe-data SFT | Fixed-q DiLoCo, but each site also trains on its accepted silver y⁺ statements; extra compute matched and reported |
| Alternative weights | (a) random weights with the same spread and box as ClinGuard's (I14); (b) loss weighting: q tilted towards sites with a higher calibration-split loss, same KL geometry and box as ClinGuard (I13) |

**Tier 2 — required if compute allows (decide at G3 from measured time)**
Pooled-data model (upper reference, simulation only); H ∈ {25, 250}; exact vs sketched scores and sketch sizes; probe-loss ablations (likelihood-only, contrast-only); with/without box constraint; IID vs skewed partition; trainable A+B LoRA; connector training.

**Tier 3 — stretch [ASK PROF Q3]**
Adaptations of FedMRG, FedIHRAS confidence weighting, MERIT-style one-shot merge, IsoLoCo. If not reproduced, state so; do not present adapted versions as the originals.

---

## 10. Evaluation

### 10.1 Frozen settings
Prompt, decoding, max length, image resolution, checkpoint rule and metric versions are frozen at G3/G4 and recorded with hashes. All methods generate on the same study lists.

### 10.2 Automatic metrics (`eval/metrics.py`)
Per site and pooled: RadGraph F1, CheXbert-label F1, BLEU-4, ROUGE-L, BERTScore; RadCliQ if it installs. Pin exact package and model versions (A7). These compare with reference text; they are **secondary** (manual §10.4).

Two reporting subsets are also shown for every metric (G1 decision):
- **without `needs_prior` studies**, whose references mention an earlier exam the model never sees;
- **"novel reference"**: evaluation studies whose cleaned reference text never occurs in that partition's training split. 28% of reports are exact templates, and a memorised template inflates BLEU and ROUGE.

### 10.3 Development proxy for the primary endpoint (`eval/proxy.py`)
The true primary endpoint needs readers, so tuning uses a pre-specified proxy: a generated report has a **proxy error** if its CheXbert labels disagree with the reference's on at least one finding labelled positive or negative in the reference (uncertain and blank handled by a fixed rule in `CODEBOOK.md`). **Worst-site proxy error rate** on development data is the selection criterion; tie-break: pooled RadGraph F1. Checkpoint rule: final round of the fixed schedule (no early-stopping search). State clearly in the report that this is a proxy.

The exact rule (`proxy_v1`) and the CheXbert reading conventions are fixed in `docs/CODEBOOK.md`. Generated reports are scored in the separate evaluation environment (I21) by `scripts/evaluate.py`.

**In CSE499A this proxy is the reported clinical endpoint (D16).** Every table and figure that uses it says "automatic proxy, not reader-assessed".

### 10.4 Primary endpoint and reader pilot (CSE499B; manual §10.2–10.3)
Runs in CSE499B, after the probe audit and the freeze, and only if qualified readers are available. If they are not, the final report states that the manual's primary endpoint was not assessed and that results rest on the automatic proxy.
- E = 1 if independent adjudication finds at least one clinically significant error in a generated report. Site rate p_s; **W = max_s p_s**; effect D = W(ClinGuard) − W(fixed-q DiLoCo), lower is better; conditioned on seed 11.
- `make_reader_pack.py`: simple random sample of **50 test studies per site** (150 total); for each, the image, the task context, and the two methods' reports in randomised A/B order; blinded key stored separately; codebook fields: false finding, omitted finding, wrong laterality/location, wrong severity, wrong negation/uncertainty, unsupported comparison, other; clinically significant yes/no; insufficient evidence.
- Two readers, independent; adjudication process fixed in advance; report agreement.
- This is a **pilot**: not powered for superiority. The confirmatory sample size is computed from pilot discordance and reported as future work unless resources appear.

### 10.5 Statistics (`eval/stats.py`; manual §11)
Report D and every site's paired difference with absolute rates; Bonferroni-adjusted binomial intervals per method × site; paired bootstrap and a paired binary test as secondary; seeds reported separately (three seeds give limited precision). One primary comparator, endpoint, direction and population, fixed before unblinding.

### 10.6 Cost metrics (`engine/accounting.py`; RQ4)
Per site and round: logical upload/download bytes (φ in fp32, sketches, summaries), GPU seconds split into train / probe / solver / eval, peak memory, wall time, unique studies and tokens consumed. Two regimes: (A) fixed training examples, probe compute reported as extra; (B) fixed total GPU time including probe and controller work. Bytes are **logical** (simulation): say so; do not claim physical network savings.

---

## 11. Experiment plan and compute budget

Primary configuration: `skew_v1`, K = 3, H = 100, T = 10, effective batch 4, seeds 11/22/33.

| Stage | Runs (estimate) | Purpose |
|---|---|---|
| S0 Profiling | a few short runs | Memory, seconds per step, seconds per generated report |
| S1 Single-site sanity | 1 | Model learns; beats zero-shot on development data |
| S2 Baseline tuning (seed 11 only) | about 9–12 | Inner lr × η_out ∈ {0.5, 0.7, 1.0} for DiLoCo; lr for FedAvg, local-only |
| S3 RQ1 sweep | 3 H values × 2 partitions × seed 11 = 6 | Does a site trade-off appear as H grows? |
| S4 ClinGuard tuning (seed 11) | about 8–12 | ρ, λ, δ |
| S5 Final Tier 1 | 8 methods × 3 seeds = 24 | Main table |
| S6 Tier 2 ablations | about 10–15, seed 11 | Ablation table |

With the **estimate** of 4–6 T4-hours per run, the total is roughly 250–400 T4-hours. The team's three Kaggle accounts, with two T4s each, give up to about 180 T4-hours per week. Runs are distributed across the accounts through `grid.yaml` (Section 4.1), and a week's work is planned so that each account's quota is used without two accounts needing the same run. **At G3, replace the estimate with measured time and cut Tier 2 first if the total exceeds the available weeks.** Equal tuning budget per method is mandatory (manual §9.1).

---

## 12. Phases, tasks and gates

Week numbers are relative (week 1 = the week of 2026-11-09) and illustrative.

**Semester targets (Q7 default).** CSE499A: G0–G4 required; G5–G6 on silver probes if time allows. CSE499B: probe audit, Phase 7, Phase 8.

### Phase 0 — Setup (week 1)
Tasks:
- Skeleton package and CI running `pytest`; `requirements.lock`; `DECISIONS.md` seeded with Section 2.
- Verify A2, A3 and A7; send the Section 2.3 questions to the supervisor; assign roles (Section 13).
- Create the HF organization (or the fine-grained-token fallback) and the private run repo. Give the other two Kaggle accounts write access, and store the tokens in Kaggle Secrets.
- Verify A9: each account can start a T4 session, clone the public repo, import the package, and write and delete a test file in the HF repo.

**Gate G0:** repo builds and tests run on a clean machine; licences recorded; metric packages import on Kaggle; all three accounts pass the A9 check.

### Phase 1 — Data (weeks 1–2)
Tasks: `ingest`, `clean_v1`, group lists, both partitions, splits, manifests, `DATA_CARD.md` (counts per site × role × group, exclusions with reasons, limitations), loader with deterministic sampler, `test_manifest.py`.
**Gate G1:** manifests hashed and frozen; no cross-split or cross-site overlap; counts table and G1 review items signed off by Tasbid; A1 and A8 resolved. After G1 the split seed and group lists never change.

### Phase 2 — Engine on toy problems (weeks 2–3)
Tasks: `SiteTask` interface; `outer.py` (D7); `weighting.py` (uniform, fixed-q, random, loss-weighted, external weights); `state.py` (atomic checkpoint: φ, m, per-site AdamW and sampler state, RNG, round id, config and manifest hashes, counters); `accounting.py`; `runner.py`; tests A.1, A.2, A.3, A.7 and `test_toy_e2e.py` (toy sites with conflicting quadratic objectives).
**Gate G2:** all engine tests pass on CPU; an interrupted-and-resumed toy run equals an uninterrupted run bit-for-bit; uniform weighting reproduces a hand-computed three-round example.

### Phase 3 — Real model integration (weeks 3–4)
Tasks:
- Model code: `load.py`, `lora.py`, `params.py`; test A.4 (LoRA identity, and a demonstration that averaging trainable A and B differs); `generate.py`.
- Profiling: `profile_model.py` (memory and time at two resolutions and microbatch sizes).
- First model results: zero-shot evaluation on development data; one single-site fine-tune; freeze prompt, resolution and decoding.
- Shared-compute tooling (Section 4.1): `utils/hf_sync.py` and `scripts/run_job.py`. They handle lease, per-round upload, resume from HF and the commit check. The checkpoint also stores CUDA RNG state. `notebooks/kaggle_runner.ipynb` is the one runbook notebook operators use. A first `grid.yaml` is written.
- Resume and handoff tests with the real model (short runs): resume on the same account; then two rounds on account 1, continued on account 2. Compare against an uninterrupted run and record the gap (A10).

**Gate G3:** a fine-tuned single-site model has fewer proxy errors than the **always-normal report** on the pooled development studies (paired, one-sided exact McNemar p < 0.05) and its reports assert findings (revised 2026-11-22, I27: in session 2 the site-A model beat zero-shot while writing one normal template); no NaNs in fp16 (or fallback chosen and logged); measured time per run recorded; compute budget table in Section 11 updated; Tier 2 scope decided; cross-account handoff works and its numerical gap is recorded. **Stop and fix if base report generation is broken** (manual §13).

### Phase 4 — Baseline pilots and RQ1 (weeks 4–6)
Tasks: local-only, FedAvg, uniform and fixed-q DiLoCo on seed 11; tuning on development data; RQ1 sweep over H on both partitions; per-site proxy error, per-site metrics and cost tables.
**Gate G4:** DiLoCo matches or beats FedAvg at equal data (otherwise sync is suspect — stop and debug); RQ1 result written up **whatever it shows**. If no site-specific trade-off appears under `skew_v1` at any H, raise it with the supervisor before investing in Phase 6 (manual: "do not manufacture one by untuning baselines").

### Phase 5 — Silver calibration set (weeks 3–5, in parallel with Phases 3–4)
Tasks:
- Write the template table `probe_templates_v1.yaml`.
- Run CheXbert labelling of the calibration reports; this needs no 3B model, so it can run locally or in a short Kaggle session. Check A11.
- Apply the two-source agreement filter (Section 7.2); freeze `probes_silver_v1`.
- Probe loss and gradient, with unit tests on the tiny VLM.
- Text-only, image-swap and paraphrase controls with the real model.
- Prepare `make_review_pack.py` for the CSE499B audit. It is not run in CSE499A.

**Gate G5:** each site has ≥ 50 accepted silver pairs (or is marked unsupported); controls show the probe depends on the image; MeSH–CheXbert agreement rates and all excluded cases logged.

### Phase 6 — ClinGuard (weeks 6–9)
Tasks: `sketch.py` + test A.5; `scores.py`; `solver.py` + test A.6 (feasible, conflicting, zero-signal, missing-group cases); reduction test (w = q and zero penalty reproduces fixed-q DiLoCo exactly, A.3); RQ2 fidelity study (exact vs sketched scores; predicted vs realised probe-loss change after the actual step); ClinGuard tuning; matched controls (probe-data SFT, random weights, loss weighting); Tier 2 ablations.
**Gate G6:** predicted and realised probe-loss changes agree in sign well above chance on development rounds; controller is not living on slack or pinned at the box every round; RQ2 written up. Apply stop rules (Section 15).

### Phase 7 — Probe audit, freeze and test (CSE499B)
Tasks:
1. Radiologist audit of the silver probes (Section 7.3). If needed, build `probes_v2` and rerun the affected runs on development data.
2. Tag the freeze commit (configs, prompts, statistics plan, reader protocol).
3. Run seeds 11/22/33 for Tier 1 if not already final.
4. Unlock the internal test split **once**; compute automatic metrics.
5. If readers are available: build and send the reader pack, import the ratings, and run the statistics.

**Gate G7:** frozen analysis executed exactly as pre-registered; audit result reported; all deviations listed.

### Phase 8 — Report and showcase (weeks 11+)
Report using the manual's Appendix B template (study ID, design, comparators, results, cost, limitations, decision). Demo for the Innovation Competition: a dashboard replaying a run — per-round weights, probe losses, per-site error proxy — and side-by-side reports on held-out images, labelled as retrospective research output, not a clinical tool.

---

## 13. Roles (suggested)

| Role | Owner | Scope |
|---|---|---|
| Project lead: all development, analysis and documents | Tasbid | Everything in `src/`, `scripts/`, `configs/`, `docs/`; the experiment grid and its account assignments; tuning; gate decisions; reports; contact with the supervisor |
| Kaggle run operators | Tazvir Rahat, Puja Ghosh (Tasbid's account is the third) | Run the jobs `grid.yaml` assigns to their account, using only `notebooks/kaggle_runner.ipynb` at the pinned commit; never edit code, configs or the HF repo by hand; after every session, report run_id, rounds completed and any error |

Tasbid writes every run, handoff and failure into `RUNLOG.md` from the operators' reports and the HF repo contents.

---

## 14. Logging and reproducibility (manual §14)

Every run writes `run.json` (code commit, config and its hash, model revision, LoRA-A checksum, manifest hash, probe-set checksum, seeds, environment lock hash, hardware) and `rounds.jsonl` (per round: per-site train loss, probe loss, weights, slack, solver status, exact and sketched scores, norms of d_i and m, bytes, seconds by category, unique studies). `docs/RUNLOG.md` gets one dated line per run: purpose, result, anomalies, resumptions, failures. Failed and abandoned runs are kept and listed.

---

## 15. Risks and stop rules

| Risk | Mitigation |
|---|---|
| IU X-ray too small: overfitting, noisy site differences | Short schedules; fixed checkpoint rule; three seeds; report uncertainty; IID control |
| No site trade-off appears (RQ1 negative) | Decide at G4 with the supervisor; a well-documented negative result is acceptable (manual §4) |
| Base model saw IU X-ray | Zero-shot baseline; state limitation; all comparisons are between methods on the same base |
| fp16 instability in 4-bit | A5 fallbacks |
| Probe gradient too noisy with 24 cases | Increase probe batch; compare with full-set gradient offline; report |
| Controller always at box or always slack | Diagnose scores; widen/narrow ρ; if it persists, report as failure condition |
| Kaggle quota or session limits | Round-level resume; two GPUs in parallel; cut Tier 2 |
| No radiologist in CSE499A (D15) | Two-source agreement filter; image-dependence controls in gate G5; results labelled "automatically derived probes"; radiologist audit before the freeze in CSE499B; if audit agreement is poor, rebuild probes and rerun on development data |
| Two accounts train the same run, or a run continues on the wrong commit or config | Lease file in the HF repo; run_id contains the config hash; resume refuses a changed config, manifest or commit |
| Token leak (public repo) | Tokens only in Kaggle Secrets; per-member or single-repo fine-grained tokens; revoke and rotate on any suspicion |
| An operator is unavailable | Any account can take over an expired lease; Tasbid's account is the fallback; plan each week's grid with slack |
| Scope creep | Tier 3 only after G6; no new features after the freeze |

**Stop or revise if (manual §13):** no site trade-off is observed; sketch predictions fail on development data; the optimizer cannot match the baseline; the controller mostly uses slack or suppresses important sites; probe compute dominates its benefit; readers find no credible error reduction. In each case, document the negative result with the conditions under which it occurred.

**Claims we will not make:** clinical safety, diagnostic validity, formal privacy, deployment readiness, real multi-hospital evidence, or physical network savings. In CSE499A also not: expert-verified probes or reader-assessed error rates.

---

## Appendix A — Unit tests (manual Appendix A; all must pass before the gate shown)

| Test | Checks | Gate |
|---|---|---|
| A.1 sign | On L(φ) = ½(φ − a)², a descent-like d lowers the loss with μ = 0 | G2 |
| A.2 momentum | Manual outer step equals `torch.optim.SGD(nesterov=True)` to machine tolerance over several rounds, including first-step initialisation; Eq. (6) verified numerically | G2 |
| A.3 reduction | w = q = 1/K with zero controller effect reproduces uniform DiLoCo exactly; fixed q equals a direct weighted sum | G2, G6 |
| A.4 LoRA identity | With common fixed A, Σ wᵢBᵢA = (Σ wᵢBᵢ)A; with trainable Aᵢ, factor-wise averaging differs from averaging products | G3 |
| A.5 sketch | Unbiased inner-product estimates over fresh seeds; variance shrinks with r; behaviour when the true signal is near zero; identical coordinate order across sites | G6 |
| A.6 constraint | Feasible, conflicting, zero-signal and missing-group cases; simplex, box and slack satisfied; predicted vs realised loss after the actual step | G6 |
| A.7 resume | Interrupted + resumed run equals an uninterrupted run (φ, m, AdamW, sampler, RNG); manifests have no cross-split overlap | G2, G3 |

## Appendix B — Example config (`configs/experiment/clinguard_primary.yaml`)

```yaml
run: {seed: 11, partition: skew_v1, sites: [A, B, C]}
model: {name: Qwen/Qwen2.5-VL-3B-Instruct, revision: <pin>, quant: nf4_4bit, compute_dtype: fp16,
        image_pixels: 200704, prompt: prompt_v1}
lora: {rank: 16, alpha: 32, dropout: 0.0, targets: [q_proj, k_proj, v_proj, o_proj],
       train: B_only, A_file: lora_A.safetensors, lora_seed: 2026}
inner: {optimizer: adamw, lr: 1.0e-4, weight_decay: 0.0, clip: 1.0, eff_batch: 4, warmup_steps: 50,
        persistent_state: true}
outer: {type: nesterov, mu: 0.9, eta: 0.7}
schedule: {H: 100, total_steps_per_site: 1000}
weighting: {type: clinguard, q: by_train_count, rho: 0.5, lambda: 10, delta: 0.0, bound_mode: relative_current,
            score_source: sketch}
probe: {set: probes_silver_v1, batch: 24, a: 1.0, b: 1.0, tau: 1.0}
sketch: {type: countsketch, r: 1024}
eval: {split: dev, decoding: greedy, max_new_tokens: 160}
ckpt: {every_rounds: 1, hf_repo: <private>}
```

## Appendix C — First project update (draft, 2026-11-11)

1. Direction confirmed: ClinGuard-DiLoCo on IU X-ray, per the methods manual; previous BatchNorm study closed.
2. Key design decisions taken (Section 2.1): three simulated sites with a controlled skew plus an IID control; FINDINGS from a frontal image; a ~3B general VLM in 4-bit with fixed-basis (B-only) LoRA; single-machine simulation with exact byte accounting; Kaggle T4 with full-state resume.
3. Implementation plan written, with eight phases and pass/fail gates, the manual's seven unit tests, and stop rules.
4. This week: repository and environment setup (Phase 0), data pipeline and frozen split manifests (Phase 1), start of the DiLoCo engine on toy problems (Phase 2).
5. Needed from the supervisor: answers to Q1–Q7 in Section 2.3 (external test set, radiologist workload and timing, baseline depth, bound definition, ethics, semester deliverable).
6. Known limitations already identified: simulated sites from one source; no patient identifiers in IU X-ray; possible base-model exposure to public data; reader study will be a pilot, not confirmatory.

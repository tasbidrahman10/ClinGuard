# Decisions log

Three kinds of entry: fixed decisions carried from the plan, assumptions and their verification
status, and implementation choices made while building. A fixed decision changes only through a dated
entry in `DEVIATIONS.md`.

## 1. Fixed decisions (plan section 2.1)

D1 to D17 are fixed as written in `IMPLEMENTATION_PLAN.md` section 2.1 (v1.1). Short form:

| ID | Decision |
|---|---|
| D1 | Dataset: IU X-ray, Kaggle copy `raddar/chest-xrays-indiana-university` |
| D2 | K = 3 simulated sites from a skewed partition, plus an IID control |
| D3 | Task: FINDINGS from one frontal image, fixed prompt |
| D4 | Base model: `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit |
| D5 | Trainable parameters: LoRA B only; LoRA A frozen and shared (rank 16) |
| D6 | Visual connector frozen in the primary configuration |
| D7 | Outer step conventions: d = phi_t - phi_local; m = mu m + d_bar; phi = phi - eta (d_bar + mu m); m starts at 0 |
| D8 | H = 100 primary, sensitivity 25 and 250; 1,000 local optimizer steps per site |
| D9 | Exact scores are the reference; sketches are the communicated approximation |
| D10 | One site-wide probe group per site in the pilot |
| D11 | Constraint bound relative to the current probe loss |
| D12 | Seeds 11 (primary), 22, 33; split seed 20261110 |
| D13 | Kaggle T4 on the team's three accounts, distributed by whole run; checkpoints in one private Hugging Face repo (plan section 4.1) |
| D14 | One config-driven Python package; no federated framework |
| D15 | CSE499A probes are silver probes: accepted only when MeSH and CheXbert-on-report agree; radiologist audit of a sample in CSE499B (added 2026-11-20) |
| D16 | CSE499A clinical endpoint is the automatic worst-site proxy; reader pilot in CSE499B (added 2026-11-20) |
| D17 | GitHub repository is public; no data-derived text or checkpoints in it (added 2026-11-20) |

## 2. Assumptions (plan section 2.2)

| ID | Assumption | Status | Evidence |
|---|---|---|---|
| A1 | Kaggle copy has the two CSV files with the expected columns, about 3,850 studies / 7,470 images | **Verified 2026-11-11 (CSV files) and 2026-11-20 (images)** | Columns match exactly; 3,851 studies, 7,466 images (3,818 frontal, 3,648 lateral). Full archive downloaded 2026-11-20: 7,470 PNG files in `images/images_normalized/`, the layout `prepare_data.py` documents for Kaggle. Both CSVs are byte-identical to the 2026-11-11 build (SHA-256 `cd725771...`, `7af9c8a6...`). |
| A2 | IU X-ray licence is CC BY-NC-ND 4.0 | **Verified 2026-11-11** | Kaggle dataset metadata: "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)", dataset version 2. Consequence unchanged: research use only; data, trained models and outputs stay private unless the supervisor approves. |
| A3 | Qwen2.5-VL-3B-Instruct licence permits research use | **Verified 2026-11-11** | Model card licence `qwen-research` (Qwen Research License Agreement): non-commercial, research or evaluation only; redistribution must carry the licence and attribution notice. Current revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (to be pinned at G3). The 7B model is Apache-2.0, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. |
| A4 | Base model was not trained on IU X-ray | Cannot be proven; limitation | Zero-shot baseline at G3 |
| A5 | Stable 4-bit training with fp16 on a T4 | Open | G3 |
| A6 | One run takes 4 to 6 T4-hours | Open (estimate) | G3 |
| A7 | `radgraph` and `f1chexbert` install on Kaggle | Open | Needs a Kaggle session |
| A8 | About 3,300 eligible studies | **Resolved: 3,199** | 514 studies have no FINDINGS text and 138 more have no frontal image (2026-11-11). Image-open check run 2026-11-20: all 3,199 chosen frontal images open, so the count is final. |
| A9 | Three team Kaggle accounts with GPU quota, each able to write to the shared HF repo | Open | G0 |
| A10 | Same environment on all accounts; GPU resume close to but not bit-identical with an uninterrupted run | Open | G3 handoff test |
| A11 | CheXbert labels of the calibration reports are reliable enough as the second silver-probe source | Open | G5 |

## 3. Questions for the supervisor (plan section 2.3)

| ID | Status |
|---|---|
| Q1 External test set | Open; default: none, stated as a limitation |
| Q2 Radiologist time | **Answered 2026-11-20** (supervisor, reported by Tasbid): no radiologist in CSE499A. Use automatic labels and automatic metrics. A radiologist may audit a sample of probes and run the reader pilot in CSE499B. Leads to D15, D16, M6, M7 |
| Q3 Baseline depth | Open; default: Tier 1 and 2 required, prior-art reproductions are stretch |
| Q4 Method name | Open; default: working label |
| Q5 Bound definition | Open; default: relative to current |
| Q6 Ethics for reader study | Deferred to CSE499B with the reader pilot |
| Q7 End of CSE499A | Open; default: G0–G4 required, G5–G6 if time allows |
| Q8 Size of the CSE499B probe audit | Open (new); default: 20 accepted silver probes per site |

## 4. Implementation choices (made while building; open to review before the gate shown)

| ID | Date | Choice | Why | Review at |
|---|---|---|---|---|
| I1 | 2026-11-11 | Site and role assignment use a hash of (seed, purpose, uid), not a running random generator | A study's site does not change when another study is added or dropped, for example after the image check | G1 |
| I2 | 2026-11-11 | In `iid_v1` every study keeps its `skew_v1` data role; only the site changes | Otherwise a study could be test data in one partition and training or tuning data in the other. Refines plan section 5.4; see `DEVIATIONS.md` | G1 |
| I3 | 2026-11-11 | Roles are cut inside each site x group at 70 / 80 / 90% of the hash-ordered list (rounded down) | Exact stratification by group, as the plan requires | G1 |
| I4 | 2026-11-11 | `groups_v1` adds five MeSH headings to the plan's wording: "airspace disease"; "calcified granuloma" and "granulomatous disease" (granuloma); "pulmonary emphysema" and "bullous emphysema" (emphysema). "subcutaneous emphysema" is not parenchymal. "cardiac shadow" and "aorta" count only with the attributes "enlarged" and "tortuous" | The plan names finding families; these are the exact MeSH spellings found in the data | G1 |
| I5 | 2026-11-11 | In `clean_v1` a token counts as de-identified when it contains `XXXX` (so "x-XXXX" is removed whole); punctuation that the removed token carried is kept | The plan does not define a token; 28 such tokens exist against 2,193 plain ones | G1 |
| I6 | 2026-11-11 | Exclusion reasons are applied in a fixed order (empty findings, too short after cleaning, no frontal image, image unreadable) and each study is counted once | Needed for an unambiguous exclusion table | G1 |
| I7 | 2026-11-11 | The per-site sampler uses its own NumPy PCG64 generator; a batch may span an epoch boundary so every study is used once per epoch | Small sites should not lose studies to a dropped last batch; the state is a plain dictionary and checkpoints exactly | G2 |
| I8 | 2026-11-11 | Logical byte accounting per site and round: download = 4 bytes x P; upload = 4 bytes x P plus a 16-byte summary; sketches are added in Phase 6 | Plan section 10.6 | G2 |
| I9 | 2026-11-11 | A checkpoint stores the config hash and manifest hash, and a run refuses to resume from a checkpoint written with different ones | Prevents silently mixing two configurations in one run | G2 |
| I10 | 2026-11-11 | Local development uses Python 3.11 with CPU PyTorch (`requirements-dev.lock`). `requirements.lock` for the GPU stack is written at G3 on the Kaggle image | The model stack cannot be pinned before it has run on Kaggle | G3 |
| I11 | 2026-11-20 | Pillow added to `requirements-dev.lock` (13.0.0) | `prepare.py` needs it for the image-open check; the check now runs locally as well as on Kaggle | G1 |
| I12 | 2026-11-20 | Shared compute protocol: whole runs per account, `LEASE.json` in each HF run folder, upload after every round, resume refused on a different code commit (extends I9) | Three accounts share one HF repo; prevents two accounts training the same run and silent mixing of code versions | G3 |
| I13 | 2026-11-20 | **Loss-weighted baseline:** w = argmax Σ wᵢ ℓᵢ/T − KL(w‖q) on the simplex inside ClinGuard's box (1 − ρ)q ≤ w ≤ (1 + ρ)q. Exact solution wᵢ = clip(qᵢ exp(ℓᵢ/T − ν), box), ν by bisection. ℓᵢ = mean loss on site i's **calibration** split at φ_t, measured before local training. Higher loss → more weight. T tuned on development data | The plan says "validation-loss weighting" without a formula. Development data may only be used for tuning and selection (manual §5.2), so the in-loop loss uses the calibration split, the data ClinGuard also sees. Same KL geometry and box as ClinGuard, so the only difference is the clinical, direction-aware signal | G4 |
| I14 | 2026-11-20 | **Random baseline:** same solution with scores σ·zᵢ, zᵢ ~ N(0, 1) from a generator seeded by (run seed, round); a fresh draw each round, no stored state. σ is set in Phase 6 to match the spread of ClinGuard's realised weights (default 0.1 until then) | "Random weights with matched variance" (plan section 9) needs a spread parameter; seeding by round keeps resume exact | G6 |

| I15 | 2026-11-20 | **G1 review decisions** (Tasbid): (1) `clean_v1` kept: dropping every XXXX sentence would remove ~14% of report words, including finding sentences; (2) `groups_v1` kept; the 82 "No Indexing" studies stay in `other_abnormal` and are never silver probes; (3) `needs_prior` word list extended to `prior_v2` (+10 words that refer to an earlier exam, +61 studies, 771 → 832); appearance words such as "increased", "old", "chronic" excluded; (4) deviation P1 approved; (5) text metrics also reported on a "novel reference" subset | Evidence in the local review sheet `data/review/G1_REVIEW.md` (not committed: quotes report text) | Frozen |
| I16 | 2026-11-20 | `prepare_data.py --check` now rebuilds every manifest file in a temporary folder and compares bytes, not only site/role assignments | The old check would not have noticed a change in the `group` or `needs_prior` columns | Frozen |

**Gate G1 passed on 2026-11-20.** Manifests frozen: `skew_v1` `fba077ca06499e73`, `iid_v1` `77e5cd83bdf165f8`.
No study changed site or role at the freeze; only the `needs_prior` column changed (prior_v2). The split
seed, group lists, cleaning rule and word list never change again; `tests/test_manifest.py` pins the hashes.

## 5. Not yet decided

- **σ for the random baseline:** matched to ClinGuard's realised weight spread once Phase 6 runs exist
  (I14). The statistic to match (standard deviation of w − q or of log(w/q)) is fixed before tuning.

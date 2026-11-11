# Decisions log

Three kinds of entry: fixed decisions carried from the plan, assumptions and their verification
status, and implementation choices made while building. A fixed decision changes only through a dated
entry in `DEVIATIONS.md`.

## 1. Fixed decisions (plan section 2.1)

D1 to D14 are fixed as written in `IMPLEMENTATION_PLAN.md` section 2.1. Short form:

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
| D13 | Kaggle T4; checkpoints to a private Hugging Face repo |
| D14 | One config-driven Python package; no federated framework |

## 2. Assumptions (plan section 2.2)

| ID | Assumption | Status | Evidence |
|---|---|---|---|
| A1 | Kaggle copy has the two CSV files with the expected columns, about 3,850 studies / 7,470 images | **Verified 2026-11-11** for the CSV files | Columns match exactly; 3,851 studies, 7,466 images (3,818 frontal, 3,648 lateral). The image folder layout on Kaggle is still to be checked. |
| A2 | IU X-ray licence is CC BY-NC-ND 4.0 | **Verified 2026-11-11** | Kaggle dataset metadata: "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)", dataset version 2. Consequence unchanged: research use only; data, trained models and outputs stay private unless the supervisor approves. |
| A3 | Qwen2.5-VL-3B-Instruct licence permits research use | **Verified 2026-11-11** | Model card licence `qwen-research` (Qwen Research License Agreement): non-commercial, research or evaluation only; redistribution must carry the licence and attribution notice. Current revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (to be pinned at G3). The 7B model is Apache-2.0, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. |
| A4 | Base model was not trained on IU X-ray | Cannot be proven; limitation | Zero-shot baseline at G3 |
| A5 | Stable 4-bit training with fp16 on a T4 | Open | G3 |
| A6 | One run takes 4 to 6 T4-hours | Open (estimate) | G3 |
| A7 | `radgraph` and `f1chexbert` install on Kaggle | Open | Needs a Kaggle session |
| A8 | About 3,300 eligible studies | **Resolved 2026-11-11: 3,199** | 514 studies have no FINDINGS text and 138 more have no frontal image. May drop slightly if any image fails to open. |

## 3. Questions for the supervisor (plan section 2.3)

Q1 to Q7 are open. Work proceeds on the stated defaults until answered.

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

## 5. Not yet decided

- **Loss-weighted baseline:** the plan names "ordinary validation-loss weighting" but gives no formula
  (which loss, which direction, what temperature). To be fixed before Phase 4 tuning.
- **Random-weight baseline:** must match the variance of ClinGuard's weights, which is unknown until
  Phase 6.

# ClinGuard-DiLoCo: Progress Update 1

**Date:** 11 November 2026
**Course:** CSE499A Senior Capstone Design, North South University
**Team:** Tasbid Al Rahman, Tazvir Rahat, Puja Ghosh
**Supervisor:** Dr. Sumaiya Tabassum Nimi

## 1. Summary

Following our last meeting, the project is now ClinGuard-DiLoCo: training one chest X-ray report
generation model across several simulated hospitals with DiLoCo, and testing whether a clinically
aware way of combining the hospitals' updates lowers the worst hospital's error rate. The earlier
BatchNorm study is closed.

This week we wrote the implementation plan, set up the code repository, built the data pipeline and
ran it on the real IU X-ray report files, and built the core DiLoCo training loop and tested it on
small synthetic problems. No model has been trained yet, and no results are claimed.

## 2. Work completed

| Area | What exists now |
|---|---|
| Plan | An implementation plan based on the methods manual: 14 fixed design decisions, eight phases with pass/fail gates, the manual's seven unit tests, and stop rules. |
| Repository | One Python package with automated tests, plus a decisions log, a deviations log and a run log. |
| Licences | Both checked. IU X-ray is CC BY-NC-ND 4.0: research use only, and we do not redistribute the data. The chosen model, Qwen2.5-VL-3B-Instruct, is under a research-only, non-commercial licence. |
| Data pipeline | Reads the IU X-ray files, applies the pre-registered eligibility and text-cleaning rules, assigns each study to a finding group, builds the three simulated hospitals, and splits each one into training, calibration, development and test sets. Every output file is hashed, and the test set cannot be loaded by the code before the analysis is frozen. |
| DiLoCo engine | The round loop: each hospital trains locally, the updates are combined with uniform or data-size weights, and the shared model takes DiLoCo's momentum step. It includes exact save-and-resume and communication-cost accounting. |
| Tests | 49 automated tests pass. They include four of the manual's seven required checks: the update-sign test, the momentum test against PyTorch's own implementation, the reduction test, and the resume test (an interrupted run gives exactly the same result as an uninterrupted one). The other three checks belong to later phases. |

## 3. The data, measured

The plan's estimates are now replaced with real counts.

| | Studies |
|---|---|
| IU X-ray studies in total | 3,851 |
| Excluded: no FINDINGS text | 514 |
| Excluded: no frontal image | 138 |
| **Eligible** | **3,199** |

The three simulated hospitals in the main (skewed) setting:

| Hospital | Studies | Normal | Heart and pleura | Lung tissue | Other |
|---|---|---|---|---|---|
| A ("screening") | 1,011 | 59% | 8% | 9% | 24% |
| B ("cardiac") | 1,050 | 25% | 36% | 14% | 24% |
| C ("pulmonary") | 1,138 | 26% | 10% | 40% | 23% |

Each hospital has about 700 to 800 training studies and about 100 to 115 studies in each of the
calibration, development and test sets. A second setting with three evenly mixed hospitals serves as
the control.

## 4. What the data showed us

- **Fewer usable studies than estimated:** 3,199 against the planned 3,300. The count may fall
  slightly once every image has been opened and checked.
- **Evaluation sets are small.** With about 100 test studies per hospital, each hospital's error
  rate will carry a wide margin of uncertainty. Hospital A has only 8 or 9 heart-and-pleura studies
  in each evaluation set.
- **The calibration sets are just large enough.** The plan needs at least 50 radiologist-approved
  probe cases per hospital, from 101 to 114 calibration studies each.
- **Many reports are templates.** 28% of eligible studies share their exact FINDINGS text with
  another study, so the same reference text can appear in training and in testing.
- **A quarter of reports refer to earlier exams.** 24% use words such as "stable" or "unchanged",
  which the model cannot judge from a single image. These stay in the data, and results will also be
  reported without them.
- **The de-identification marks affect the text.** 42% of eligible reports contained hidden words.
  Removing them, as pre-registered, sometimes leaves a broken phrase or a slightly stronger
  statement. We will review this before freezing the data.

## 5. What is not done yet

- No gate in the plan has been declared passed.
- The split files are candidates, not frozen. The images (14 GB) have not been checked yet; this
  will run on Kaggle.
- The model has not been loaded or trained. That is the next phase after the data is frozen.
- Two baseline weighting rules (random and loss-based) are not yet in the engine.
- The private code and checkpoint repositories are still to be created.

## 6. Plan for next week

1. Check every image on Kaggle, review the counts as a team, and freeze the data splits.
2. Confirm the evaluation packages (RadGraph, CheXbert) install on Kaggle.
3. Finish the remaining engine pieces and close the engine gate.
4. Load the model on a Kaggle T4, measure memory and time per step, and run the untrained baseline.
5. Start writing the candidate probe cases for radiologist review.

## 7. Decisions we need from you

Work continues on the default shown for each until we hear otherwise.

| # | Question | Our default |
|---|---|---|
| 1 | IU X-ray is now the training data, so there is no separate external test set. Is that acceptable? | No external set; stated as a limitation |
| 2 | Can a radiologist review about 150 probe cases (50 per hospital) within about five weeks, and can two readers assess about 150 cases with two reports each near the end? | Plan on these numbers |
| 3 | How far should we go in reproducing earlier methods (FedMRG, FedIHRAS, MERIT, IsoLoCo)? | Core baselines required; these four only if time allows |
| 4 | "ClinGuard" is already the name of an unrelated paper. Should we rename now? | Keep it as a working label |
| 5 | Should a hospital's safety limit be set against the current model or against a fixed early checkpoint? | Against the current model |
| 6 | Is ethics approval needed for the reader study on public, de-identified data? | We wait for your guidance before that phase |
| 7 | What must be finished by the end of CSE499A? | The first five gates: setup, data, engine, model integration, baselines |

## 8. Known limitations

- The three hospitals are simulated from one institution's data; they are not real hospitals.
- IU X-ray identifies studies, not patients, so two exams of one patient could fall in different sets.
- The model may have seen IU X-ray during its original training, since the data is public.
- The reader study will be a pilot. It is not large enough to prove that one method is better.

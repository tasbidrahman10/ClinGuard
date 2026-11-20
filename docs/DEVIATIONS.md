# Deviations log

Every departure from the methods manual or from the implementation plan: what, why, who approved.

## From the methods manual (recorded in the plan, 2026-11-10)

| # | Manual | What we do instead | Why | Approved by |
|---|---|---|---|---|
| M1 | Section 5.1: MIMIC-CXR and CheXpert Plus as sites, IU X-ray as external test | IU X-ray is the only source; three simulated sites; no external test set | Supervisor's instruction; access and compute | Supervisor (dataset); external test set is question Q1 |
| M2 | Section 9.2: LoRA plus a fixed list of connector parameters | Connector frozen in the primary configuration; trained only as an ablation | Keeps the trainable vector linear in B and the 4-bit setup simple (D6) | Pending |
| M3 | Section 9.2: inner learning rates 1e-5, 5e-5, 1e-4 | Grid 5e-5, 1e-4, 2e-4 | B-only LoRA usually needs a higher rate; to be confirmed by tuning | Pending |
| M4 | Section 6.4: bound fixed from a reference checkpoint | Bound relative to the current probe loss (D11); the manual's version stays as a config option | A fixed early reference becomes permissive as the model improves | Question Q5 |
| M5 | Section 5.1: assign patients to splits | Studies are the unit | IU X-ray has no patient identifier | Limitation; cannot be fixed |
| M6 | Sections 6.2 and 8: a radiologist verifies every probe before use | CSE499A uses silver probes, accepted only when MeSH and CheXbert-on-report labels agree (plan 7.2). A radiologist audits a stratified sample in CSE499B, before the freeze (plan 7.3). Results are labelled "automatically derived probes, not expert-verified" | No radiologist available in CSE499A | Supervisor, 2026-11-20 |
| M7 | Sections 10.2–10.3: primary endpoint from blinded, adjudicated reader assessment | CSE499A reports the automatic worst-site proxy (plan 10.3) with CheXbert and RadGraph metrics on development data. The reader pilot moves to CSE499B and runs only if readers are available | No radiologist available in CSE499A | Supervisor, 2026-11-20 |

## From the implementation plan

| # | Date | Plan | What we do instead | Why | Approved by |
|---|---|---|---|---|---|
| P1 | 2026-11-11 | Section 5.4: `iid_v1` is a uniform random site assignment; section 5.5 then splits each site 70 / 10 / 10 / 10 | Every study keeps the data role it has in `skew_v1`; within each role x group, studies are dealt evenly to the three sites in a seeded random order | With independent splits, about 90% of each partition's test studies would be training, calibration or development data in the other partition. Site sizes and group mix in `iid_v1` are equal by construction | **Approved by Tasbid at G1, 2026-11-20** |
| P2 | 2026-11-20 | Sections 4 and 12: code in a private GitHub repo | The GitHub repo is public (D17). It holds code, configs, docs and uid-only manifests; checkpoints and data-derived text stay in the private HF repo | Kaggle notebooks clone without a token; the manifests hold no report text or images | Tasbid (project lead) |
| P3 | 2026-11-20 | Section 13: three development roles | Tasbid leads all development; the other two members operate Kaggle runs only | Workload distribution within the team | Tasbid (project lead) |

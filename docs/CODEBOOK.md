# Codebook: automatic labels and the proxy endpoint

Plan sections 10.3 and 7.2. This file fixes how automatic labels are read. A change needs a new version name
and a dated entry in `DEVIATIONS.md`; results always state the version used.

## CheXbert labels

- Labeller: CheXbert as packaged in `f1chexbert==0.0.2`, checkpoint `chexbert.pth` from
  `StanfordAIMI/RRG_scorers` at revision `6646433b3ad83a10f6e141db76d0ece44312b236`, run in the evaluation
  environment (`requirements-eval.lock`, transformers 4.46.3; decision I21).
- Mode `classification`: for each of 14 classes the label is 1 (positive), 0 (negative), -1 (uncertain)
  or blank (not mentioned).
- Classes: Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation,
  Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices,
  No Finding.

## Proxy error, version `proxy_v1`

A generated report has a proxy error when, for at least one of the 13 findings other than "No Finding":

| Reference label | Generated label | Proxy error? | Kind |
|---|---|---|---|
| positive | positive or uncertain | no | |
| positive | negative | **yes** | contradicted |
| positive | blank | **yes** | omitted |
| negative | positive or uncertain | **yes** | false finding |
| negative | negative or blank | no | (omitting a negative is not an error) |
| uncertain or blank | anything | no | no reliable truth to compare with |

An uncertain generated statement counts as asserting the finding, which is the usual report-generation
convention.

**Site rate** = share of a site's development reports with a proxy error. The **worst-site proxy error
rate** = the maximum over the participating sites. This rate is the CSE499A selection criterion and the
CSE499A clinical endpoint (D16). Every table using it says **"automatic proxy, not reader-assessed"**.

## Known limits of the proxy

- CheXbert reads text; it never sees the image. A reference that is wrong stays wrong.
- "Not mentioned" in the reference is never treated as absent.
- A report can avoid proxy errors by mentioning nothing positive in a study whose reference is all
  negative or blank. Pair the proxy with CheXbert F1 and RadGraph, and watch report length.

## CheXbert F1

Each label is turned into a binary label: positive or uncertain = 1, otherwise 0. F1 is reported micro and
macro over all 14 classes and over the 5 standard classes (Cardiomegaly, Edema, Consolidation, Atelectasis,
Pleural Effusion).

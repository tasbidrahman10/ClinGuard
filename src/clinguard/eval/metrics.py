"""Automatic evaluation of generated FINDINGS (plan sections 10.2 and 10.3).

Runs in the evaluation environment (I21: transformers 4.x, f1chexbert, radgraph, CPU). The scoring logic
below is plain Python so it can be unit-tested anywhere; the two labeller models are loaded lazily.

Per report:
  - CheXbert labels of the generated and the reference text (14 classes; 1 positive, 0 negative,
    -1 uncertain, '' not mentioned).
  - Proxy error, rule proxy_v1 (docs/CODEBOOK.md): over the 13 findings other than "No Finding",
      reference positive -> error unless the generated report is positive or uncertain (omission / contradiction)
      reference negative -> error if the generated report is positive or uncertain (false finding)
      reference uncertain or not mentioned -> never an error (no reliable truth to compare with)
  - RadGraph-XL rewards (simple, partial, complete).
  - ROUGE-L F (beta 1.2, as in pycocoevalcap) on lower-cased word tokens.
Per group of reports: proxy error rate, CheXbert F1 (micro and macro, 14 and 5 classes; uncertain counts as
positive, the usual report-generation convention), mean RadGraph rewards, corpus BLEU-4, mean ROUGE-L.

Two baselines (decision I32): "always normal" (asserts no finding) and the best constant report
CONSTANT_REPORT_V1, the one fixed set of findings that, asserted for every study, makes the fewest proxy_v1
errors on the CALIBRATION reports. proxy_v1 counts a false finding only where the reference rules the finding
out, and reports rarely rule out atelectasis, pleural thickening or devices, so naming them for everyone is
nearly free: that report beats always-normal on the development studies (12 studies fixed, 1 broken,
p = 0.002) without looking at a single image.

Every group is reported for three subsets (G1 decision): all reports, reports whose reference does not
need a prior exam, and "novel reference" reports whose reference text never occurs in training.
"""
import math
import re
from collections import Counter

PROXY_VERSION = "proxy_v1"
CHEXBERT_CLASSES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
                    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                    "Pleural Other", "Fracture", "Support Devices", "No Finding"]
FIVE_CLASSES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
POS, NEG, UNC, BLANK = 1, 0, -1, ""
_FINDINGS = [c for c in CHEXBERT_CLASSES if c != "No Finding"]
# Best constant report: searched over every set of up to 3 findings on the 320 calibration reports
# (CheXbert labels, labels/chexbert_calibration.json), fewest proxy_v1 errors, ties to the smaller set.
# Calibration 0.322 vs always-normal 0.350; development 0.306 vs 0.341. Fixed on 2026-11-26.
CONSTANT_REPORT_V1 = ("Atelectasis", "Pleural Other", "Support Devices")


# ------------------------------------------------------------------------------------ per-report rules

def proxy_errors(ref, hyp):
    """[(class, kind)] of proxy_v1 errors for one report; empty list = no proxy error."""
    out = []
    for c in _FINDINGS:
        i = CHEXBERT_CLASSES.index(c)
        r, h = ref[i], hyp[i]
        said = h in (POS, UNC)
        if r == POS and not said:
            out.append((c, "omitted" if h == BLANK else "contradicted"))
        elif r == NEG and said:
            out.append((c, "false_finding"))
    return out


def always_normal_error(ref):
    """proxy_v1 error of a report that asserts no finding (the "always normal" baseline): an error exactly
    when the reference has a positive finding (omitted or contradicted; a false finding is impossible)."""
    return any(ref[CHEXBERT_CLASSES.index(c)] == POS for c in _FINDINGS)


def label_report(findings):
    """CheXbert labels of a report that asserts exactly `findings` and mentions nothing else."""
    unknown = set(findings) - set(_FINDINGS)
    if unknown:
        raise ValueError(f"unknown findings {sorted(unknown)}")
    return [POS if c in findings else BLANK for c in CHEXBERT_CLASSES]


def constant_report_error(ref, findings=CONSTANT_REPORT_V1):
    """proxy_v1 error of the constant report that asserts `findings` for every study."""
    return bool(proxy_errors(ref, label_report(findings)))


def label_level_check(predicted, ref_labels):
    """proxy_v1 result if each study's report asserted exactly its predicted findings and nothing else (a
    writer that copies its diagnosis prompt perfectly). predicted: {uid: [findings]}; ref_labels: {uid: 14 labels}."""
    uids = [u for u in predicted if u in ref_labels]
    model = [bool(proxy_errors(ref_labels[u], label_report(predicted[u]))) for u in uids]
    normal = [always_normal_error(ref_labels[u]) for u in uids]
    const = [constant_report_error(ref_labels[u]) for u in uids]
    n = len(uids)
    return dict(n=n, error_rate=sum(model) / n, always_normal_error_rate=sum(normal) / n,
                constant_report_error_rate=sum(const) / n, studies_with_a_hint=sum(bool(predicted[u]) for u in uids),
                vs_always_normal=mcnemar_one_sided(model, normal), vs_constant_report=mcnemar_one_sided(model, const))


def asserts_finding(labels):
    return any(labels[CHEXBERT_CLASSES.index(c)] in (POS, UNC) for c in _FINDINGS)


def mcnemar_one_sided(model_errors, baseline_errors):
    """Paired exact McNemar test that the model makes FEWER errors than the baseline on the same reports.
    b = model wrong, baseline right; c = model right, baseline wrong; p = P(X <= b), X ~ Binomial(b + c, 1/2)."""
    b = sum(m and not s for m, s in zip(model_errors, baseline_errors))
    c = sum(s and not m for m, s in zip(model_errors, baseline_errors))
    n = b + c
    p = sum(math.comb(n, k) for k in range(b + 1)) / 2 ** n if n else 1.0
    return dict(model_only_errors=b, baseline_only_errors=c, p_one_sided=p)


def binary(labels):
    """Report-generation convention: positive or uncertain -> 1, otherwise 0."""
    return [1 if v in (POS, UNC) else 0 for v in labels]


def f1_scores(refs, hyps, classes=CHEXBERT_CLASSES):
    """Micro and macro F1 over the given classes, from lists of 14-label vectors."""
    idx = [CHEXBERT_CLASSES.index(c) for c in classes]
    tp, fp, fn = Counter(), Counter(), Counter()
    for r, h in zip(refs, hyps):
        rb, hb = binary(r), binary(h)
        for i in idx:
            tp[i] += rb[i] and hb[i]
            fp[i] += (not rb[i]) and hb[i]
            fn[i] += rb[i] and not hb[i]

    def f1(t, p, n):
        return 2 * t / (2 * t + p + n) if (2 * t + p + n) else 0.0
    macro = sum(f1(tp[i], fp[i], fn[i]) for i in idx) / len(idx)
    micro = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return dict(micro=micro, macro=macro)


_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[^\sa-z0-9]")


def tokens(text):
    return _TOKEN.findall(text.lower())


def rouge_l(ref, hyp, beta=1.2):
    r, h = tokens(ref), tokens(hyp)
    if not r or not h:
        return 0.0
    prev = [0] * (len(h) + 1)
    for a in r:
        cur = [0]
        for j, b in enumerate(h):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return (1 + beta ** 2) * p * rc / (rc + beta ** 2 * p)


def corpus_bleu4(refs, hyps):
    """Corpus BLEU-4 (one reference per report, uniform weights, brevity penalty, no smoothing)."""
    match, total = [0] * 4, [0] * 4
    ref_len = hyp_len = 0
    for ref, hyp in zip(refs, hyps):
        r, h = tokens(ref), tokens(hyp)
        ref_len, hyp_len = ref_len + len(r), hyp_len + len(h)
        for n in range(1, 5):
            hn = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    if hyp_len == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(log_p)


# ------------------------------------------------------------------------------------------ aggregation

def score_reports(rows, chexbert_label, radgraph_rewards):
    """Adds per-report scores to each row. chexbert_label(text) -> 14 labels;
    radgraph_rewards(refs, hyps) -> list of (simple, partial, complete) per report."""
    rg = radgraph_rewards([r["reference"] for r in rows], [r["generated"] for r in rows])
    out = []
    for r, rewards in zip(rows, rg):
        ref_l, hyp_l = chexbert_label(r["reference"]), chexbert_label(r["generated"])
        errs = proxy_errors(ref_l, hyp_l)
        out.append({**r, "ref_labels": ref_l, "hyp_labels": hyp_l, "proxy_error": bool(errs), "proxy_details": errs,
                    "radgraph": list(rewards), "rouge_l": rouge_l(r["reference"], r["generated"])})
    return out


def group_metrics(scored):
    if not scored:
        return dict(n=0)
    refs = [s["ref_labels"] for s in scored]
    hyps = [s["hyp_labels"] for s in scored]
    n = len(scored)
    texts = Counter(s["generated"].strip() for s in scored)
    finding_idx = [CHEXBERT_CLASSES.index(c) for c in _FINDINGS]
    positives = [(r, h, i) for r, h in zip(refs, hyps) for i in finding_idx if r[i] == POS]
    recalled = sum(h[i] in (POS, UNC) for _, h, i in positives)
    return dict(
        n=n,
        proxy_error_rate=sum(s["proxy_error"] for s in scored) / n,
        proxy_errors=sum(s["proxy_error"] for s in scored),
        always_normal_error_rate=sum(always_normal_error(r) for r in refs) / n,
        constant_report_error_rate=sum(constant_report_error(r) for r in refs) / n,
        share_asserting_a_finding=sum(asserts_finding(h) for h in hyps) / n,
        finding_recall=recalled / len(positives) if positives else None,
        distinct_text_share=len(texts) / n,
        top_text_share=texts.most_common(1)[0][1] / n,
        chexbert_f1_14=f1_scores(refs, hyps),
        chexbert_f1_5=f1_scores(refs, hyps, FIVE_CLASSES),
        radgraph=dict(zip(("simple", "partial", "complete"),
                          (sum(s["radgraph"][k] for s in scored) / n for k in range(3)))),
        bleu4=corpus_bleu4([s["reference"] for s in scored], [s["generated"] for s in scored]),
        rouge_l=sum(s["rouge_l"] for s in scored) / n,
        mean_words_generated=sum(len(s["generated"].split()) for s in scored) / n,
        mean_words_reference=sum(len(s["reference"].split()) for s in scored) / n,
    )


def summarize(scored, training_texts):
    """Per site, pooled and worst site, for each subset."""
    subsets = {
        "all": scored,
        "no_prior": [s for s in scored if not s["needs_prior"]],
        "novel_reference": [s for s in scored if s["reference"] not in training_texts],
    }
    out = {}
    for name, rows in subsets.items():
        sites = sorted({s["site"] for s in rows})
        per_site = {site: group_metrics([s for s in rows if s["site"] == site]) for site in sites}
        rated = {site: m["proxy_error_rate"] for site, m in per_site.items() if m["n"]}
        worst = max(rated, key=rated.get) if rated else None
        normal = {site: m["always_normal_error_rate"] for site, m in per_site.items() if m["n"]}
        out[name] = dict(sites=per_site, pooled=group_metrics(rows),
                         worst_site=dict(site=worst, proxy_error_rate=rated.get(worst),
                                         always_normal_worst=max(normal.values()) if normal else None),
                         vs_always_normal=mcnemar_one_sided([s["proxy_error"] for s in rows],
                                                            [always_normal_error(s["ref_labels"]) for s in rows]),
                         vs_constant_report=mcnemar_one_sided([s["proxy_error"] for s in rows],
                                                              [constant_report_error(s["ref_labels"]) for s in rows]))
    return dict(proxy_version=PROXY_VERSION, constant_report=list(CONSTANT_REPORT_V1), subsets=out)


# ------------------------------------------------------------------------------------------- labellers

CHEXBERT_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"          # StanfordAIMI/RRG_scorers


def load_scorers(device="cpu"):
    """(chexbert_label, radgraph_rewards) from the pinned scorer checkpoints. Evaluation environment only."""
    import os
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise RuntimeError("CheXbert and RadGraph need transformers 4.x: use the evaluation environment (I21)")
    import f1chexbert.f1chexbert as fm
    from huggingface_hub import hf_hub_download
    from radgraph import F1RadGraph
    if not os.path.exists(os.path.join(fm.CACHE_DIR, "chexbert.pth")):
        hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth", revision=CHEXBERT_REVISION, local_dir=fm.CACHE_DIR)
    cx = fm.F1CheXbert(device=device)
    if cx.target_names != CHEXBERT_CLASSES:
        raise RuntimeError("unexpected CheXbert class order")
    rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    def chexbert_label(text):
        return cx.get_label(text.strip() or ".", mode="classification")

    def radgraph_rewards(refs, hyps):
        _, per_report, _, _ = rg(refs=refs, hyps=[h if h.strip() else "." for h in hyps])
        return list(zip(*per_report))
    return chexbert_label, radgraph_rewards

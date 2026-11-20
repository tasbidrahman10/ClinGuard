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
    return dict(
        n=n,
        proxy_error_rate=sum(s["proxy_error"] for s in scored) / n,
        proxy_errors=sum(s["proxy_error"] for s in scored),
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
        out[name] = dict(sites=per_site, pooled=group_metrics(rows),
                         worst_site=dict(site=worst, proxy_error_rate=rated.get(worst)))
    return dict(proxy_version=PROXY_VERSION, subsets=out)


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

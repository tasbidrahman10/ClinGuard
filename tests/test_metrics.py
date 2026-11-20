"""Evaluation logic (plan 10.2-10.3, docs/CODEBOOK.md). Unit tests run anywhere; the last test needs the
evaluation environment (real CheXbert and RadGraph, transformers 4.x) and skips elsewhere."""
import math

import pytest

from clinguard.eval import metrics as M

C = M.CHEXBERT_CLASSES


def labels(**kw):
    v = [M.BLANK] * len(C)
    for name, val in kw.items():
        v[C.index(name.replace("_", " "))] = val
    return v


@pytest.mark.parametrize("ref,hyp,expected", [
    (M.POS, M.POS, []), (M.POS, M.UNC, []),
    (M.POS, M.NEG, [("Pleural Effusion", "contradicted")]),
    (M.POS, M.BLANK, [("Pleural Effusion", "omitted")]),
    (M.NEG, M.POS, [("Pleural Effusion", "false_finding")]),
    (M.NEG, M.UNC, [("Pleural Effusion", "false_finding")]),
    (M.NEG, M.NEG, []), (M.NEG, M.BLANK, []),
    (M.UNC, M.NEG, []), (M.BLANK, M.POS, []),
])
def test_proxy_v1_truth_table(ref, hyp, expected):
    assert M.proxy_errors(labels(Pleural_Effusion=ref), labels(Pleural_Effusion=hyp)) == expected


def test_no_finding_never_counts():
    assert M.proxy_errors(labels(No_Finding=M.POS), labels(No_Finding=M.BLANK)) == []


def test_f1_counts():
    refs = [labels(Cardiomegaly=M.POS), labels(Cardiomegaly=M.POS, Edema=M.UNC), labels()]
    hyps = [labels(Cardiomegaly=M.POS), labels(Edema=M.POS), labels(Cardiomegaly=M.NEG, Edema=M.POS)]
    # Cardiomegaly: tp 1, fn 1; Edema: tp 1, fp 1  -> micro F1 = 2*2 / (2*2 + 1 + 1) = 2/3
    f = M.f1_scores(refs, hyps)
    assert f["micro"] == pytest.approx(2 / 3)
    five = M.f1_scores(refs, hyps, M.FIVE_CLASSES)
    assert five["macro"] == pytest.approx((2 / 3 + 2 / 3) / 5)          # three of five classes have F1 = 0


def test_text_metrics():
    assert M.rouge_l("The lungs are clear.", "the lungs are clear .") == pytest.approx(1.0)
    assert M.rouge_l("heart normal", "pleural effusion") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P = R = 3/4 -> F = 3/4
    assert M.rouge_l("a b c d", "a c d e") == pytest.approx(0.75)
    same = ["no pleural effusion or pneumothorax .", "heart size is normal ."]
    assert M.corpus_bleu4(same, same) == pytest.approx(1.0)
    assert M.corpus_bleu4(["heart size is normal"], ["lungs clear"]) == 0.0
    assert M.tokens("Size 16.2 cm, stable.") == ["size", "16.2", "cm", ",", "stable", "."]


def _rows():
    def row(site, uid, ref, hyp, prior=False):
        return dict(site=site, uid=uid, group="normal", needs_prior=prior, reference=ref, generated=hyp)
    return [row("A", "1", "effusion", "effusion"), row("A", "2", "effusion", "clear"),
            row("B", "3", "clear", "clear"), row("B", "4", "clear", "clear", prior=True),
            row("B", "5", "template", "effusion")]


def _fake_scorers():
    table = {"effusion": labels(Pleural_Effusion=M.POS), "clear": labels(Pleural_Effusion=M.NEG, No_Finding=M.POS),
             "template": labels(Pleural_Effusion=M.NEG)}
    return (lambda t: table[t]), (lambda refs, hyps: [(1.0, 0.5, 0.25)] * len(refs))


def test_summary_sites_subsets_and_worst_site():
    chexbert, radgraph = _fake_scorers()
    scored = M.score_reports(_rows(), chexbert, radgraph)
    assert [s["proxy_error"] for s in scored] == [False, True, False, False, True]
    summ = M.summarize(scored, training_texts={"template"})
    allsub = summ["subsets"]["all"]
    assert allsub["sites"]["A"]["proxy_error_rate"] == 0.5
    assert allsub["sites"]["B"]["proxy_error_rate"] == pytest.approx(1 / 3)
    assert allsub["worst_site"] == dict(site="A", proxy_error_rate=0.5)
    assert allsub["pooled"]["n"] == 5 and allsub["pooled"]["radgraph"]["partial"] == 0.5
    assert summ["subsets"]["no_prior"]["sites"]["B"]["n"] == 2
    novel = summ["subsets"]["novel_reference"]["sites"]["B"]
    assert novel["n"] == 2 and novel["proxy_error_rate"] == 0.0           # the templated reference is excluded
    assert summ["proxy_version"] == "proxy_v1"


def test_real_scorers_in_the_evaluation_environment():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("f1chexbert")
    pytest.importorskip("radgraph")
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("evaluation environment only (transformers 4.x)")
    nltk_bleu = pytest.importorskip("nltk.translate.bleu_score")
    chexbert, radgraph = M.load_scorers()
    eff = chexbert("There is a small left pleural effusion. Heart size is normal.")
    assert eff[C.index("Pleural Effusion")] == M.POS and eff[C.index("Cardiomegaly")] == M.NEG
    assert chexbert("The lungs are clear. No acute cardiopulmonary abnormality.")[C.index("No Finding")] == M.POS
    rewards = radgraph(["There is a small left pleural effusion.", "The lungs are clear."],
                       ["Small left pleural effusion is present.", "Right lower lobe opacity."])
    assert len(rewards) == 2 and all(len(r) == 3 for r in rewards) and rewards[0][1] > rewards[1][1]

    refs = ["no pleural effusion or pneumothorax .", "heart size is normal and the lungs are clear ."]
    hyps = ["no pleural effusion .", "the heart size is normal , lungs are clear ."]
    ours = M.corpus_bleu4(refs, hyps)
    theirs = nltk_bleu.corpus_bleu([[M.tokens(r)] for r in refs], [M.tokens(h) for h in hyps])
    assert math.isclose(ours, theirs, rel_tol=1e-9)

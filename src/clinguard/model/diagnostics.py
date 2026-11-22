"""Does the model use the image? (gate G3, Kaggle session 3)

Teacher-forced loss of each study's reference FINDINGS under three image conditions:
  true      the study's own frontal image
  shuffled  the image of another study in the same list (the list rotated by half its length)
  blank     a uniform mid-grey image
A model that reads the image predicts its own study's report better with the true image than with another
study's image, most clearly on abnormal studies. A model that ignores the image has the same loss under all
three conditions. Only numbers leave this module; no report text.
"""
import contextlib
import dataclasses

import torch
import torch.nn.functional as F

from clinguard.data.dataset import IGNORE_INDEX, batch_to, build_batch

CONDITIONS = ("true", "shuffled", "blank")


def write_blank_image(path, size=448):
    from PIL import Image
    Image.new("RGB", (size, size), (128, 128, 128)).save(path)


def with_images(examples, condition, blank_path=None):
    """The same studies and reports with the images replaced according to `condition`."""
    n = len(examples)
    if condition == "true":
        return list(examples)
    if condition == "shuffled":
        if n < 2:
            raise ValueError("the shuffled condition needs at least two studies")
        k = n // 2
        return [dataclasses.replace(e, image_path=examples[(i + k) % n].image_path) for i, e in enumerate(examples)]
    if condition == "blank":
        if not blank_path:
            raise ValueError("the blank condition needs blank_path")
        return [dataclasses.replace(e, image_path=blank_path) for e in examples]
    raise ValueError(f"unknown condition '{condition}'")


def per_example_losses(model, processor, examples, amp_dtype=None, batch_size=4):
    """[(summed cross-entropy over the answer tokens, number of answer tokens)] per example, in order.
    Eval mode, no gradients; the model's training flag is restored."""
    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    use_amp = amp_dtype is not None and device.type == "cuda"
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                enc = batch_to(build_batch(processor, examples[i:i + batch_size]), device)
                labels = enc.pop("labels")
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                with ctx:
                    logits = model(**enc).logits
                for r in range(logits.shape[0]):
                    target = labels[r, 1:]
                    keep = target != IGNORE_INDEX
                    loss = F.cross_entropy(logits[r, :-1][keep].float(), target[keep], reduction="sum")
                    out.append((loss.item(), int(keep.sum())))
    finally:
        model.train(was_training)
    return out


def summarize(examples, losses):
    """losses: {condition: per_example_losses output}. Token-weighted loss per condition, and the per-study
    comparison of true vs shuffled image, for all / normal / abnormal studies."""
    subsets = {"all": lambda e: True, "normal": lambda e: e.group == "normal",
               "abnormal": lambda e: e.group != "normal"}
    out = {}
    for name, keep in subsets.items():
        idx = [i for i, e in enumerate(examples) if keep(e)]
        d = dict(n=len(idx))
        for c, rows in losses.items():
            tokens = sum(rows[i][1] for i in idx)
            d[f"loss_{c}"] = sum(rows[i][0] for i in idx) / tokens if tokens else None
        if idx and "true" in losses and "shuffled" in losses:
            pairs = [(losses["true"][i][0] / losses["true"][i][1], losses["shuffled"][i][0] / losses["shuffled"][i][1])
                     for i in idx]
            d["share_true_image_better"] = sum(t < s for t, s in pairs) / len(pairs)
            d["mean_gain_true_vs_shuffled"] = sum(s - t for t, s in pairs) / len(pairs)
        out[name] = d
    return out



# ------------------------------------------------------------------ finding probes (vision check, session 4)
# For each finding: a positive and a negative statement in report style, and which reference CheXbert labels
# count as "absent" (an explicit negative; for devices, which radiologists rarely rule out, any non-positive).
PROBE_STATEMENTS = {
    "Cardiomegaly": ("The heart is enlarged.", "The heart size is normal.", "negative"),
    "Lung Opacity": ("There is an airspace opacity in the lung.", "The lungs are clear.", "negative"),
    "Pleural Effusion": ("There is a pleural effusion.", "There is no pleural effusion.", "negative"),
    "Support Devices": ("A pacemaker or other medical device is present.", "There are no medical devices.",
                        "not_positive"),
}


def probe_scores(model, processor, examples, positive, negative, amp_dtype=None, batch_size=8):
    """Per study: log P(positive statement | image) - log P(negative statement | image), teacher-forced as
    the assistant's answer. The statements are the same for every study, so any difference between studies
    comes from the image alone."""
    pos = per_example_losses(model, processor, [dataclasses.replace(e, findings=positive) for e in examples],
                             amp_dtype, batch_size)
    neg = per_example_losses(model, processor, [dataclasses.replace(e, findings=negative) for e in examples],
                             amp_dtype, batch_size)
    return [n[0] - p[0] for p, n in zip(pos, neg)]


def auroc(positive_scores, negative_scores):
    """Probability that a random present-finding study scores above a random absent one (ties count 1/2)."""
    if not positive_scores or not negative_scores:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive_scores for n in negative_scores)
    return wins / (len(positive_scores) * len(negative_scores))


def finding_probe_check(model, processor, examples, ref_labels, amp_dtype=None, batch_size=8):
    """{finding: dict(n_present, n_absent, auroc)}. ref_labels: {uid: 14 CheXbert labels of the reference}."""
    from clinguard.eval.metrics import CHEXBERT_CLASSES, POS, NEG
    out = {}
    for finding, (positive, negative, absent_rule) in PROBE_STATEMENTS.items():
        i = CHEXBERT_CLASSES.index(finding)
        present = [e for e in examples if e.uid in ref_labels and ref_labels[e.uid][i] == POS]
        if absent_rule == "negative":
            absent = [e for e in examples if e.uid in ref_labels and ref_labels[e.uid][i] == NEG]
        else:
            absent = [e for e in examples if e.uid in ref_labels and ref_labels[e.uid][i] != POS]
        scores = probe_scores(model, processor, present + absent, positive, negative, amp_dtype, batch_size)
        out[finding] = dict(n_present=len(present), n_absent=len(absent),
                            auroc=auroc(scores[:len(present)], scores[len(present):]))
    return out


def image_test(model, processor, examples, blank_path, amp_dtype=None, batch_size=4):
    losses = {c: per_example_losses(model, processor, with_images(examples, c, blank_path), amp_dtype, batch_size)
              for c in CONDITIONS}
    return summarize(examples, losses)

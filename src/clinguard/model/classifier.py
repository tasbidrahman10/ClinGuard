"""Finding classifier for the diagnosis prompt (decision I32, Kaggle session 7).

The report writer never learned to read findings off the image (sessions 1-6), so a separate classifier
names them and the writer gets them as words in its prompt (the diagnosis prompt of FedMRG / PromptMRG).

  1. Image features: each study's frontal image goes once through the vision tower and merger (with the
     taught vision stage merged in); the image tokens the language model would receive are pooled into the
     mean and the maximum of every channel (2 x hidden numbers per study).
  2. One logistic regression per finding (dataset.HINT_WORDS), trained on the TRAINING studies with
     CheXbert labels of their reports: positive = 1; negative or not mentioned = 0; uncertain left out.
     Features standardized with the training mean and spread; the L2 strength is picked from L2_GRID by
     log loss on the calibration studies.
  3. A cut-off per finding on the CALIBRATION studies: the score above which the most studies are flagged
     correctly minus incorrectly. A flag is incorrect whenever the report does not state the finding,
     whether it rules it out or does not mention it, so naming a finding pays off only where it is more
     often right than wrong. If no cut-off flags more studies correctly than incorrectly, the finding is
     never named (cut-off None).
Deterministic: float64 L-BFGS on the CPU from zero weights.
"""
import contextlib

import torch
import torch.nn.functional as F

from clinguard.data.dataset import HINT_WORDS, load_image

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _get_image_features(model):
    """The model's get_image_features, also through peft wrappers."""
    fn = getattr(model, "get_image_features", None)
    if fn is None:
        raise ValueError("the model has no get_image_features")
    return fn


@torch.no_grad()
def image_features(model, processor, examples, batch_size=16, amp_dtype=None):
    """[n, 2 * hidden] float32 tensor: mean and max over each study's image tokens (merger output)."""
    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    use_amp = amp_dtype is not None and device.type == "cuda"
    get = _get_image_features(model)
    was_training = model.training
    model.eval()
    out = []
    try:
        for i in range(0, len(examples), batch_size):
            images = [load_image(e.image_path) for e in examples[i:i + batch_size]]
            enc = processor.image_processor(images=images, return_tensors="pt")
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with ctx:
                feats = get(enc["pixel_values"].to(device), enc["image_grid_thw"].to(device))
            per_image = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            if len(per_image) != len(images):
                raise RuntimeError(f"{len(per_image)} feature blocks for {len(images)} images")
            out += [torch.cat([f.float().mean(0), f.float().amax(0)]).cpu() for f in per_image]
    finally:
        model.train(was_training)
    return torch.stack(out)


def target(labels, finding):
    """1 positive, 0 negative or not mentioned, None uncertain (left out)."""
    from clinguard.eval.metrics import CHEXBERT_CLASSES, POS, UNC
    v = labels[CHEXBERT_CLASSES.index(finding)]
    return None if v == UNC else int(v == POS)


def true_hint(labels, findings):
    """The findings among `findings` that a report's CheXbert labels mark positive, in HINT_WORDS order."""
    from clinguard.eval.metrics import CHEXBERT_CLASSES, POS
    return tuple(f for f in HINT_WORDS if f in findings and labels[CHEXBERT_CLASSES.index(f)] == POS)


def fit_logistic(x, y, l2, max_iter=500):
    """(w, b) minimizing mean log loss + l2 / 2 * |w|^2 on standardized features x [n, d], y in {0, 1}."""
    x, y = x.double(), y.double()
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    prior = y.mean().clamp(1e-6, 1 - 1e-6)
    b = torch.log(prior / (1 - prior)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def log_loss(scores, y):
    return F.binary_cross_entropy_with_logits(scores.double(), y.double()).item()


def auroc(scores, y):
    """Probability that a random positive scores above a random negative (ties count 1/2); None if one class is empty."""
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def choose_cutoff(scores, y):
    """dict(cutoff, net, correct, wrong): flag studies with score >= cutoff, where `cutoff` maximizes
    correct - wrong flags (ties: fewer flags). cutoff None when no cut-off makes net > 0."""
    order = sorted(zip(scores, y), key=lambda t: -t[0])
    best = dict(cutoff=None, net=0, correct=0, wrong=0)
    correct = wrong = 0
    for i, (s, t) in enumerate(order):
        correct, wrong = correct + bool(t), wrong + (not t)
        if i + 1 < len(order) and order[i + 1][0] == s:          # cut only between different scores
            continue
        if correct - wrong > best["net"]:
            best = dict(cutoff=float(s), net=correct - wrong, correct=correct, wrong=wrong)
    return best


def _rows(uids, feats, labels, finding):
    keep = [(i, target(labels[u], finding)) for i, u in enumerate(uids) if u in labels]
    keep = [(i, t) for i, t in keep if t is not None]
    return [i for i, _ in keep], torch.tensor([t for _, t in keep], dtype=torch.float64)


def fit_classifiers(train, calib, findings=tuple(HINT_WORDS), l2_grid=L2_GRID):
    """train / calib: (uids, features [n, d], {uid: 14 CheXbert labels}). Returns {finding: model dict}
    with the standardization, weights, chosen L2, cut-off and calibration numbers."""
    t_uids, t_x, t_lab = train
    c_uids, c_x, c_lab = calib
    mean, std = t_x.double().mean(0), t_x.double().std(0) + 1e-6
    zt, zc = (t_x.double() - mean) / std, (c_x.double() - mean) / std
    models = {}
    for finding in findings:
        ti, ty = _rows(t_uids, t_x, t_lab, finding)
        ci, cy = _rows(c_uids, c_x, c_lab, finding)
        if ty.sum() == 0 or ty.sum() == len(ty):
            raise ValueError(f"{finding}: the training studies have only one class")
        fits = []
        for l2 in l2_grid:
            w, b = fit_logistic(zt[ti], ty, l2)
            fits.append((log_loss(zc[ci] @ w + b, cy), l2, w, b))
        loss, l2, w, b = min(fits, key=lambda f: (f[0], -f[1]))      # ties: the stronger penalty
        cal_scores = (zc[ci] @ w + b).tolist()
        models[finding] = dict(w=w, b=b, l2=l2, calib_log_loss=loss, **choose_cutoff(cal_scores, cy.tolist()),
                               train_positive=int(ty.sum()), train_n=len(ty),
                               calib_positive=int(cy.sum()), calib_n=len(cy),
                               calib_auroc=auroc(cal_scores, cy.tolist()))
    return dict(mean=mean, std=std, models=models)


def scores(clf, x):
    """{finding: [score per row of x]} (log-odds)."""
    z = (x.double() - clf["mean"]) / clf["std"]
    return {f: (z @ m["w"] + m["b"]).tolist() for f, m in clf["models"].items()}


def predict(clf, uids, x):
    """{uid: [findings at or above their cut-off]}, findings in HINT_WORDS order."""
    s = scores(clf, x)
    out = {}
    for k, u in enumerate(uids):
        out[u] = [f for f, m in clf["models"].items() if m["cutoff"] is not None and s[f][k] >= m["cutoff"]]
    return out


def summary(clf, dev=None):
    """Numbers only, per finding; with dev = (uids, features, labels) also AUROC and flags on development."""
    out = {}
    dev_scores = scores(clf, dev[1]) if dev else None
    for f, m in clf["models"].items():
        row = {k: m[k] for k in ("train_positive", "train_n", "calib_positive", "calib_n", "l2", "calib_log_loss",
                                 "calib_auroc", "cutoff", "net", "correct", "wrong")}
        if dev:
            idx, y = _rows(dev[0], dev[1], dev[2], f)
            sc = [dev_scores[f][i] for i in idx]
            flags = [m["cutoff"] is not None and v >= m["cutoff"] for v in sc]
            row.update(dev_positive=int(y.sum()), dev_n=len(y), dev_auroc=auroc(sc, y.tolist()),
                       dev_correct=sum(bool(fl and t) for fl, t in zip(flags, y.tolist())),
                       dev_wrong=sum(bool(fl and not t) for fl, t in zip(flags, y.tolist())))
        out[f] = row
    return out

"""Finding-sentence training objective (decision I30, Kaggle session 5: "teach the eyes").

Instead of writing the whole report, the model sees an X-ray and two sentences about one finding, for example
"The heart is enlarged." and "The heart size is normal.", and is trained to give the correct one the higher
probability:   loss = -log sigmoid( log P(correct | image) - log P(wrong | image) ).

The training pairs come from CheXbert labels of the TRAINING reports (scripts/label_reports.py). For every
finding, studies with and without it are balanced (absent studies sub-sampled to at most `max_ratio` times
the present ones, present ones repeated to match), so always preferring "normal" gains nothing: only reading
the image lowers the loss. Same statements as the vision check (diagnostics.PROBE_STATEMENTS) plus a few
more findings; the vision check then measures the skill on development patients the model never saw.
"""
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from clinguard.data.dataset import IGNORE_INDEX, Example, batch_to, build_batch
from clinguard.model.diagnostics import PROBE_STATEMENTS
from clinguard.model.site import VLMSite

# PROBE_STATEMENTS (the four findings of the vision check) plus findings with enough training cases.
TRAIN_STATEMENTS = {
    **PROBE_STATEMENTS,
    "Enlarged Cardiomediastinum": ("The cardiomediastinal silhouette is enlarged.",
                                   "The cardiomediastinal silhouette is normal.", "negative"),
    "Atelectasis": ("There is atelectasis.", "There is no atelectasis.", "negative"),
    "Edema": ("There is pulmonary edema.", "There is no pulmonary edema.", "negative"),
    "Consolidation": ("There is a focal consolidation.", "There is no focal consolidation.", "negative"),
    "Pneumothorax": ("There is a pneumothorax.", "There is no pneumothorax.", "negative"),
    "Lung Lesion": ("There is a lung nodule or mass.", "There is no lung nodule or mass.", "negative"),
    "Fracture": ("There is a fracture.", "There is no fracture.", "negative"),
}


@dataclass(frozen=True)
class FindingPair:
    uid: str
    image_path: str
    finding: str
    correct: str
    wrong: str
    present: bool


def finding_pairs(examples, labels, statements=TRAIN_STATEMENTS, min_count=20, max_ratio=4, seed=0):
    """Balanced (image, correct sentence, wrong sentence) pairs and a per-finding count table.
    labels: {uid: 14 CheXbert labels}. Findings with fewer than `min_count` present or absent studies are
    skipped. Deterministic for a given seed."""
    from clinguard.eval.metrics import CHEXBERT_CLASSES, NEG, POS
    rng = random.Random(seed)
    pairs, table = [], {}
    for finding, (positive, negative, absent_rule) in statements.items():
        i = CHEXBERT_CLASSES.index(finding)
        present = [e for e in examples if e.uid in labels and labels[e.uid][i] == POS]
        if absent_rule == "negative":
            absent = [e for e in examples if e.uid in labels and labels[e.uid][i] == NEG]
        else:
            absent = [e for e in examples if e.uid in labels and labels[e.uid][i] != POS]
        if len(present) < min_count or len(absent) < min_count:
            table[finding] = dict(present=len(present), absent=len(absent), used=False)
            continue
        if len(absent) > max_ratio * len(present):
            absent = rng.sample(absent, max_ratio * len(present))
        if len(present) > max_ratio * len(absent):
            present = rng.sample(present, max_ratio * len(absent))
        rep_p = max(1, round(len(absent) / len(present)))
        rep_a = max(1, round(len(present) / len(absent)))
        pairs += [FindingPair(e.uid, e.image_path, finding, positive, negative, True) for e in present] * rep_p
        pairs += [FindingPair(e.uid, e.image_path, finding, negative, positive, False) for e in absent] * rep_a
        table[finding] = dict(present=len(present), absent=len(absent), repeat_present=rep_p,
                              repeat_absent=rep_a, used=True)
    if not pairs:
        raise ValueError("no finding has enough labelled studies")
    return pairs, table


def _as_example(pair, text):
    return Example(uid=pair.uid, image_path=pair.image_path, findings=text, group="", needs_prior=False)


def pair_logprob_gap(model, processor, pairs, device, autocast):
    """Per pair: log P(correct | image) - log P(wrong | image), summed over the sentence tokens, with
    gradients. Both sentences of all pairs go through the model in one batch."""
    enc = batch_to(build_batch(processor, [_as_example(p, p.correct) for p in pairs]
                               + [_as_example(p, p.wrong) for p in pairs]), device)
    labels = enc.pop("labels")
    with autocast:
        logits = model(**enc).logits
    target = labels[:, 1:]
    keep = target != IGNORE_INDEX
    rows = keep.nonzero()[:, 0]
    nll = F.cross_entropy(logits[:, :-1][keep].float(), target[keep], reduction="none")
    seq_nll = torch.zeros(logits.shape[0], device=nll.device, dtype=nll.dtype).index_add(0, rows, nll)
    n = len(pairs)
    return seq_nll[n:] - seq_nll[:n]


class FindingSite(VLMSite):
    """A site whose training examples are FindingPairs and whose loss is the pairwise sentence loss.
    Everything else (optimizer, sampler, warm-up, GradScaler, checkpointing) is VLMSite's."""

    def _loss(self, batch):
        gap = pair_logprob_gap(self.model, self.processor, batch, self.device, self._autocast())
        return F.softplus(-gap).mean()

"""Examples and batches for the report-generation model (plan section 5.7).

An example is one study: its frontal image and its cleaned FINDINGS text. Examples are built only from
FROZEN manifests, and the raw CSV files are checked against the hashes recorded in the manifest, so every
machine (laptop, desktop, any Kaggle account) trains on byte-identical data.

Prompt `prompt_v1` (frozen at G3):
  system: "You are an expert chest radiologist."
  user:   <image> + "Write the FINDINGS section of the radiology report for this frontal chest X-ray."
Target: the cleaned FINDINGS. The loss covers only the assistant's answer tokens.

Diagnosis prompt (decision I32, Kaggle session 7): an example may carry a `hint`, a tuple of findings
(CheXbert class names from HINT_WORDS). The user turn then ends with one more sentence,
  "Findings suggested by an image classifier: cardiomegaly, pleural effusion."   (or "...: none.")
In training the hint lists the report's own findings; at generation it lists a classifier's predictions.
Without a hint the prompt is exactly prompt_v1.
"""
import functools
import os
from dataclasses import dataclass

import torch

from clinguard.data import clean, ingest, manifest
from clinguard.utils.hashing import sha256_file, sha256_text

PROMPT_VERSION = "prompt_v1"
SYSTEM_PROMPT = "You are an expert chest radiologist."
USER_PROMPT = "Write the FINDINGS section of the radiology report for this frontal chest X-ray."
IGNORE_INDEX = -100
HINT_PROMPT_VERSION = "prompt_v2_hint"
HINT_PREFIX = "Findings suggested by an image classifier: "
# Findings a hint can name (CheXbert classes with at least 30 positive training reports), in this order, and
# the words used for them in the prompt.
HINT_WORDS = {
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinal silhouette",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung opacity",
    "Lung Lesion": "lung nodule or mass",
    "Atelectasis": "atelectasis",
    "Pleural Effusion": "pleural effusion",
    "Fracture": "fracture",
    "Support Devices": "medical device",
}


@dataclass(frozen=True)
class Example:
    uid: str
    image_path: str
    findings: str
    group: str
    needs_prior: bool
    hint: tuple = None          # findings named in the prompt (diagnosis prompt, I32); None = prompt_v1


@functools.lru_cache(maxsize=4)
def _cleaned_text(raw_dir):
    """{uid: cleaned findings} for every study in the raw files (cached per folder)."""
    return {s.uid: clean.clean_findings(s.findings_raw) for s in ingest.load_studies(raw_dir)}


def verify_source(raw_dir, manifest_dir, partition):
    """Raises unless the manifests are frozen and the raw files and cleaned text match the manifest."""
    m = manifest.read_manifest(manifest_dir, partition)
    if m.get("status") != "frozen":
        raise RuntimeError(f"{partition} manifests are not frozen; training needs gate G1")
    for fname, expected in m["raw_sha256"].items():
        if sha256_file(os.path.join(raw_dir, fname)) != expected:
            raise RuntimeError(f"{fname} differs from the file the manifests were built from")
    texts = _cleaned_text(raw_dir)
    uids = sorted(manifest.verify_partition(manifest_dir, partition), key=int)
    if sha256_text("\n".join(f"{u}\t{texts[u]}" for u in uids)) != m["cleaned_text_sha256"]:
        raise RuntimeError("cleaned text differs from the frozen build (different cleaning code?)")
    return m


def load_examples(raw_dir, images_dir, manifest_dir, partition, site, role, unlock_test=False, freeze_tag=None):
    """Examples of one site x role, in manifest order. The test split stays locked (manifest.load_split)."""
    rows = manifest.load_split(manifest_dir, partition, site, role, unlock_test=unlock_test, freeze_tag=freeze_tag)
    texts = _cleaned_text(raw_dir)
    return [Example(uid=r["uid"], image_path=os.path.join(images_dir, r["image"]), findings=texts[r["uid"]],
                    group=r["group"], needs_prior=r["needs_prior"] == "1") for r in rows]


def load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def hint_text(hint):
    """The sentence a hint adds to the prompt; findings in HINT_WORDS order."""
    unknown = set(hint) - set(HINT_WORDS)
    if unknown:
        raise ValueError(f"findings without hint words: {sorted(unknown)}")
    words = [w for f, w in HINT_WORDS.items() if f in hint]
    return HINT_PREFIX + (", ".join(words) if words else "none") + "."


def user_prompt(hint=None):
    return USER_PROMPT if hint is None else f"{USER_PROMPT} {hint_text(hint)}"


def messages(answer=None, hint=None):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt(hint)}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def _answer_start(ids, marker):
    """Index just after the last occurrence of `marker` (the assistant header) in a 1-D id tensor."""
    n, m = ids.numel(), len(marker)
    for start in range(n - m, -1, -1):
        if ids[start:start + m].tolist() == marker:
            return start + m
    raise ValueError("assistant header not found in the tokenized conversation")


def build_batch(processor, examples, with_targets=True):
    """Tokenized batch. Training batches are right-padded and carry `labels` that supervise only the
    answer (the FINDINGS text and the end-of-turn token). Generation batches are left-padded prompts."""
    images = [load_image(e.image_path) for e in examples]
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "right" if with_targets else "left"
    try:
        if with_targets:
            texts = [processor.apply_chat_template(messages(e.findings, e.hint), tokenize=False) for e in examples]
        else:
            texts = [processor.apply_chat_template(messages(hint=e.hint), tokenize=False, add_generation_prompt=True)
                     for e in examples]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = old_side
    if with_targets:
        header = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
        marker = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        if not header.endswith("<|im_start|>assistant\n"):
            raise ValueError("unexpected chat template: the generation prompt does not end with the assistant header")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = IGNORE_INDEX
        for row in range(labels.shape[0]):
            labels[row, :_answer_start(enc["input_ids"][row], marker)] = IGNORE_INDEX
        enc["labels"] = labels
    return enc


def batch_to(enc, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}

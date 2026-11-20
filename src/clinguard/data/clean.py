"""Text cleaning for the FINDINGS section, version clean_v1 (pre-registered, plan section 5.3).

Rules:
  1. Collapse whitespace. Original casing and punctuation are kept.
  2. IU de-identification writes XXXX. A sentence is removed when more than half of its tokens
     contain XXXX; otherwise every token containing XXXX is deleted and spacing is tidied.
  3. needs_prior is True when the cleaned text matches a fixed word list (version PRIOR_VERSION). Such
     studies stay in training and testing (references are not edited); results are also reported
     without them, and they never become probes.

prior_v1 (pre-registered): the first seven words below. prior_v2 (adopted at gate G1, 2026-11-20) adds
words that refer to an earlier exam; words that describe appearance ("increased", "old", "chronic")
are deliberately left out, since one image can show them. A trailing * matches any word ending.

Cleaned text is derived on the fly and is never written over the raw data or redistributed.
"""
import re

CLEAN_VERSION = "clean_v1"
PRIOR_VERSION = "prior_v2"
MIN_WORDS = 5
MASK = "XXXX"
PRIOR_WORDS = ["compared", "prior", "previous", "unchanged", "stable", "interval", "again",          # prior_v1
               "previously", "redemonstrat*", "persist*", "remain", "remains", "remained",           # prior_v2
               "resolved", "no change", "since", "now", "comparison"]


def _word_pattern(w):
    body = r"\s+".join(re.escape(part) for part in w.rstrip("*").split())
    return body + (r"\w*" if w.endswith("*") else "")


_PRIOR_RE = re.compile(r"\b(" + "|".join(_word_pattern(w) for w in PRIOR_WORDS) + r")\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")


def split_sentences(text):
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _clean_sentence(sentence):
    tokens = sentence.split()
    masked = [MASK in t for t in tokens]
    if not tokens or sum(masked) * 2 > len(tokens):
        return ""
    kept = []
    for tok, is_masked in zip(tokens, masked):
        if not is_masked:
            kept.append(tok)
            continue
        tail = _TRAILING_PUNCT.search(tok)
        if tail and kept:                                   # keep the punctuation the token carried
            kept[-1] = _TRAILING_PUNCT.sub("", kept[-1]) + tail.group()
    out = " ".join(kept)
    return out if re.search(r"[A-Za-z]", out) else ""


def clean_findings(raw):
    text = " ".join((raw or "").split())
    return " ".join(s for s in (_clean_sentence(x) for x in split_sentences(text)) if s)


def needs_prior(cleaned):
    return bool(_PRIOR_RE.search(cleaned))


def word_count(cleaned):
    return len(cleaned.split())

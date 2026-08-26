"""Cluster the fail reasons across a run.

Judges write short phrases, not codes, so the same problem shows up as "does
not compile", "compile error", "won't build". This groups them into buckets so
you can see what your generator gets wrong most often. It's deliberately simple:
a handful of keyword buckets plus a catch-all. Edit BUCKETS for your task.
"""

from __future__ import annotations

import re
from collections import Counter

from judge_jury.jury import PairResult


# (bucket label, keywords). First matching bucket wins.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("does not compile / runtime error", ("compile", "syntax", "runtime", "crash", "won't run", "wont run", "exception", "traceback")),
    ("drifts from the reference",         ("drift", "contradic", "mismatch", "does not match", "inconsistent", "disagree")),
    ("invented API or fact",              ("invent", "hallucin", "made up", "made-up", "nonexistent", "does not exist", "fabricat")),
    ("missing edge case",                 ("edge case", "empty", "null", "boundary", "off-by-one", "overflow")),
    ("incomplete",                        ("incomplete", "missing", "partial", "truncat", "stub", "todo")),
    ("style / clarity",                   ("style", "unclear", "readab", "naming", "convention", "format")),
]


def bucket_of(reason: str) -> str:
    low = reason.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def cluster_fail_reasons(results: list[PairResult], top: int = 10) -> list[tuple[str, int]]:
    """Count reason buckets over every fail verdict in the run."""
    counts: Counter[str] = Counter()
    for r in results:
        if r.outcome != "fail":
            continue
        for v in r.verdicts:
            if v.ok and v.verdict == "fail":
                for reason in v.reasons:
                    counts[bucket_of(reason)] += 1
        if r.arbiter and r.arbiter.ok and r.arbiter.verdict == "fail":
            for reason in r.arbiter.reasons:
                counts[bucket_of(reason)] += 1
    return counts.most_common(top)


def raw_fail_phrases(results: list[PairResult], top: int = 20) -> list[tuple[str, int]]:
    """The literal phrases, normalized lightly. Useful for tuning BUCKETS."""
    counts: Counter[str] = Counter()
    for r in results:
        if r.outcome != "fail":
            continue
        for v in r.verdicts:
            if v.ok and v.verdict == "fail":
                for reason in v.reasons:
                    counts[_normalize(reason)] += 1
    return counts.most_common(top)


def _normalize(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase.strip().lower())

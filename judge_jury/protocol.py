"""The verdict schema every judge answers in, plus a tolerant parser.

Free-text reviews can't be counted, so each judge is forced into one JSON
object. The parser is deliberately forgiving about the wrapping (models like to
add ```json fences or a sentence of preamble) but strict about the fields it
keeps. A verdict it can't read becomes an error upstream, not a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


# The axes are opinionated. Swap them for your task; the jury logic doesn't
# care what the axes are, only that every judge uses the same ones.
SCORE_AXES = ("correctness", "completeness", "grounding")


VERDICT_INSTRUCTIONS = f"""You are one independent reviewer on a panel. Judge the pair below on its own merits. Do not try to be lenient or harsh to match anyone else.

Return exactly one JSON object and nothing else:

{{
  "verdict": "pass" | "fail",
  "confidence": a number from 0 to 1,
  "fatal": true | false,
  "scores": {{ "correctness": 0-4, "completeness": 0-4, "grounding": 0-4 }},
  "reasons": ["short phrase", ...],
  "needs_check": ["anything you could not verify yourself", ...]
}}

Rules:
- "fatal" is true when the response is broken in a way scores don't capture: it will not run, it invents an API or fact, or it contradicts the provided context. A fatal pair fails no matter how it scores.
- "grounding" is how well the response matches the provided context. If no context is given, score grounding 4 and say so in reasons.
- Keep "reasons" to short phrases, not paragraphs. They get clustered across the dataset.
- Judge only what is in front of you. Put anything you'd need to run or look up into "needs_check"."""


@dataclass
class Verdict:
    """One judge's answer for one pair. `raw` keeps the original text for audit."""

    verdict: str                      # "pass" | "fail"
    confidence: float = 0.5
    fatal: bool = False
    scores: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    needs_check: list[str] = field(default_factory=list)
    judge: str = ""                   # filled in by the runner
    error: str | None = None          # set when the judge failed to answer
    raw: str = ""

    @property
    def ok(self) -> bool:
        """True when this verdict can be counted (the judge actually answered)."""
        return self.error is None and self.verdict in ("pass", "fail")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    """Pull the first plausible JSON object out of a model reply."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None
    # Walk braces so a trailing sentence after the object doesn't break parsing.
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_verdict(value: Any) -> str:
    v = str(value).strip().lower()
    if v.startswith("pass") or v in ("ok", "good", "accept", "yes"):
        return "pass"
    if v.startswith("fail") or v in ("bad", "reject", "no"):
        return "fail"
    return ""


def parse_verdict(text: str, judge: str = "") -> Verdict:
    """Turn a raw judge reply into a Verdict. Never raises.

    An unreadable reply comes back as a Verdict with `error` set, so the caller
    treats it like any other judge failure instead of crashing the run.
    """
    blob = _extract_json_object(text or "")
    if blob is None:
        return Verdict(verdict="", judge=judge, error="no JSON object in reply", raw=text or "")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return Verdict(verdict="", judge=judge, error=f"bad JSON: {exc}", raw=text or "")

    verdict = _coerce_verdict(data.get("verdict", ""))
    if not verdict:
        return Verdict(verdict="", judge=judge, error="missing verdict field", raw=text or "")

    scores = {}
    raw_scores = data.get("scores") or {}
    if isinstance(raw_scores, dict):
        for axis in SCORE_AXES:
            try:
                scores[axis] = max(0, min(4, int(raw_scores.get(axis, 0))))
            except (TypeError, ValueError):
                scores[axis] = 0

    def _as_list(x: Any) -> list[str]:
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return Verdict(
        verdict=verdict,
        confidence=confidence,
        fatal=bool(data.get("fatal", False)),
        scores=scores,
        reasons=_as_list(data.get("reasons")),
        needs_check=_as_list(data.get("needs_check")),
        judge=judge,
        raw=text or "",
    )

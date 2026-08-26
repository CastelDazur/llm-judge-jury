"""A fake judge for the offline demo and for CI.

It never calls a network. It reads the pair, applies a few crude heuristics, and
returns a real Verdict object, so you can run the whole jury end to end without
standing up three model servers first. Each mock judge is seeded differently so
they disagree, which is the whole point of the demo.

This is not a judge you would ship a dataset with. It exists so `judge-jury
demo` produces real-looking output in two seconds.
"""

from __future__ import annotations

import hashlib
from typing import Any

from judge_jury.backends import Judge
from judge_jury.protocol import Verdict


_BROKEN = ("does not compile", "syntaxerror", "todo", "raise notimplemented", "# broken", "undefined")
_DRIFT = ("ignores", "wrong reference", "contradicts")


class MockJudge(Judge):
    """A Judge whose `ask` is replaced by local heuristics. Bias shifts its threshold."""

    def __init__(self, name: str, bias: float = 0.0):
        super().__init__(name=name, base_url="mock://", model="heuristic")
        self.bias = bias      # >0 stricter, <0 softer

    def ask(self, system: str, user: str) -> str:  # noqa: D401 - overrides network call
        low = user.lower()
        signal = 0.0
        reasons: list[str] = []
        fatal = False

        for kw in _BROKEN:
            if kw in low:
                signal += 1.0
                fatal = True
                reasons.append("does not compile / runtime error")
                break
        for kw in _DRIFT:
            if kw in low:
                signal += 0.6
                reasons.append("drifts from the reference")
                break
        if "todo" in low or "stub" in low:
            signal += 0.4
            reasons.append("incomplete")

        # A little deterministic jitter per (judge, pair) so judges split on
        # borderline pairs instead of always agreeing.
        h = int(hashlib.sha1(f"{self.name}:{user}".encode()).hexdigest(), 16)
        jitter = ((h % 100) / 100.0 - 0.5) * 0.6
        score = signal + jitter + self.bias

        verdict = "fail" if score >= 0.5 else "pass"
        if verdict == "pass":
            reasons = ["looks correct"]
            fatal = False
        confidence = round(min(0.99, 0.55 + abs(score - 0.5)), 2)

        axes = {"correctness": 1 if verdict == "fail" else 3,
                "completeness": 2 if "incomplete" in reasons else 3,
                "grounding": 2 if "drifts from the reference" in reasons else 4}

        import json
        return json.dumps({
            "verdict": verdict,
            "confidence": confidence,
            "fatal": fatal,
            "scores": axes,
            "reasons": reasons,
            "needs_check": [],
        })


def demo_jury() -> tuple[list[MockJudge], MockJudge]:
    """Four mock judges with different strictness plus an arbiter.

    Four (an even panel) is deliberate for the demo: borderline pairs can split
    2-2, which is what sends a pair to the arbiter. With three judges a tie is
    impossible and you never see that path.
    """
    judges = [
        MockJudge("mock-strict", bias=0.25),
        MockJudge("mock-mid-a", bias=0.05),
        MockJudge("mock-mid-b", bias=-0.05),
        MockJudge("mock-soft", bias=-0.25),
    ]
    arbiter = MockJudge("mock-arbiter", bias=0.10)
    return judges, arbiter

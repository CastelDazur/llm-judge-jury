"""Run-level numbers: outcome tally, per-judge strictness, pairwise agreement.

The strictness table is the one people underrate. If a judge fails 3% of
everything while the others fail 20%, it is rubber-stamping and its vote is
close to worthless. If one fails 60%, it may be miscalibrated for your task.
You want judges that fail at similar rates but disagree on *which* pairs, that
is what independence looks like.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from judge_jury.jury import PairResult


@dataclass
class JudgeStats:
    name: str
    answered: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0

    @property
    def fail_rate(self) -> float:
        return self.failed / self.answered if self.answered else 0.0


def outcome_tally(results: list[PairResult]) -> dict[str, int]:
    tally = Counter(r.outcome for r in results)
    # Ensure all keys exist for stable reporting.
    for k in ("pass", "fail", "conflict", "pending"):
        tally.setdefault(k, 0)
    return dict(tally)


def arbiter_tally(results: list[PairResult]) -> dict[str, int]:
    """How the arbiter split the conflicts it was handed."""
    t = Counter()
    for r in results:
        if r.arbiter is not None and r.arbiter.ok:
            t[r.arbiter.verdict] += 1
    return {"pass": t.get("pass", 0), "fail": t.get("fail", 0)}


def judge_stats(results: list[PairResult]) -> list[JudgeStats]:
    stats: dict[str, JudgeStats] = defaultdict(lambda: JudgeStats(name=""))
    for r in results:
        for v in r.verdicts:
            s = stats[v.judge]
            s.name = v.judge
            if v.ok:
                s.answered += 1
                if v.verdict == "pass":
                    s.passed += 1
                else:
                    s.failed += 1
            else:
                s.errored += 1
    return sorted(stats.values(), key=lambda s: s.fail_rate, reverse=True)


def pairwise_agreement(results: list[PairResult]) -> dict[tuple[str, str], float]:
    """Fraction of pairs where two judges gave the same verdict, over pairs both judged."""
    same: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    for r in results:
        answered = [v for v in r.verdicts if v.ok]
        for i in range(len(answered)):
            for k in range(i + 1, len(answered)):
                a, b = sorted((answered[i].judge, answered[k].judge))
                total[(a, b)] += 1
                if answered[i].verdict == answered[k].verdict:
                    same[(a, b)] += 1
    return {pair: same[pair] / total[pair] for pair in total if total[pair]}

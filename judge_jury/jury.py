"""The jury: run each pair past every judge, then decide.

Decision rules, in order:
  1. If any judge that answered flags `fatal`, the pair fails. One credible
     "this is broken" outweighs a soft majority of "looks fine".
  2. Otherwise count pass/fail over the judges that answered.
       - clear majority        -> pass or fail
       - tie / no majority      -> conflict, sent to the arbiter
  3. If too few judges answered (all errored), the pair is `pending`, not a
     silent drop. Re-run it.

The arbiter is just another judge, used only to break conflicts so a tie isn't
settled by a coin. Its vote is final for that pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from judge_jury.backends import Judge, JudgeError
from judge_jury.protocol import Verdict, parse_verdict, VERDICT_INSTRUCTIONS


@dataclass
class PairResult:
    pair_id: str
    outcome: str                       # pass | fail | conflict | pending
    verdicts: list[Verdict] = field(default_factory=list)
    arbiter: Verdict | None = None
    n_pass: int = 0
    n_fail: int = 0
    n_error: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "outcome": self.outcome,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_error": self.n_error,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "arbiter": self.arbiter.to_dict() if self.arbiter else None,
        }


def build_user_prompt(pair: dict[str, Any]) -> str:
    """Render one pair into the text a judge sees."""
    parts = [
        "INSTRUCTION:",
        str(pair.get("instruction", "")).strip(),
        "",
        "RESPONSE:",
        str(pair.get("response", "")).strip(),
    ]
    context = pair.get("context")
    if context:
        import json

        parts += ["", "CONTEXT (grounding to check against):", json.dumps(context, ensure_ascii=False, indent=2)]
    return "\n".join(parts)


def judge_pair(pair: dict[str, Any], judges: list[Judge]) -> list[Verdict]:
    """Collect one verdict per judge for a single pair. Failures become error verdicts."""
    user = build_user_prompt(pair)
    out: list[Verdict] = []
    for j in judges:
        try:
            reply = j.ask(VERDICT_INSTRUCTIONS, user)
            out.append(parse_verdict(reply, judge=j.name))
        except JudgeError as exc:
            out.append(Verdict(verdict="", judge=j.name, error=str(exc)))
    return out


def decide(
    pair_id: str,
    verdicts: list[Verdict],
    arbiter: Judge | None = None,
    pair: dict[str, Any] | None = None,
    pass_rule: str = "majority",
    min_voters: int = 2,
) -> PairResult:
    """Turn a set of verdicts into one outcome, using the arbiter on conflicts."""
    counted = [v for v in verdicts if v.ok]
    n_error = len(verdicts) - len(counted)
    n_pass = sum(1 for v in counted if v.verdict == "pass")
    n_fail = sum(1 for v in counted if v.verdict == "fail")

    result = PairResult(pair_id=pair_id, outcome="pending", verdicts=verdicts,
                        n_pass=n_pass, n_fail=n_fail, n_error=n_error)

    if len(counted) < min_voters:
        result.outcome = "pending"     # not enough real votes to trust anything
        return result

    # A credible fatal flag ends it. Scores can look fine on a broken pair.
    if any(v.fatal for v in counted):
        result.outcome = "fail"
        return result

    if _passes(n_pass, n_fail, len(counted), pass_rule):
        result.outcome = "pass"
    elif _fails(n_pass, n_fail, len(counted), pass_rule):
        result.outcome = "fail"
    else:
        result.outcome = "conflict"
        if arbiter is not None and pair is not None:
            result.arbiter = _run_arbiter(arbiter, pair)
            if result.arbiter.ok:
                result.outcome = result.arbiter.verdict
    return result


def _passes(n_pass: int, n_fail: int, total: int, rule: str) -> bool:
    if rule == "unanimous":
        return n_pass == total
    if rule.startswith("k="):
        return n_pass >= int(rule[2:])
    return n_pass > n_fail          # majority


def _fails(n_pass: int, n_fail: int, total: int, rule: str) -> bool:
    if rule == "unanimous":
        return n_fail == total
    if rule.startswith("k="):
        return n_fail >= int(rule[2:])
    return n_fail > n_pass          # majority


def _run_arbiter(arbiter: Judge, pair: dict[str, Any]) -> Verdict:
    user = build_user_prompt(pair)
    try:
        reply = arbiter.ask(VERDICT_INSTRUCTIONS, user)
        return parse_verdict(reply, judge=f"{arbiter.name}(arbiter)")
    except JudgeError as exc:
        return Verdict(verdict="", judge=f"{arbiter.name}(arbiter)", error=str(exc))


def run_jury(
    pairs: list[dict[str, Any]],
    judges: list[Judge],
    arbiter: Judge | None = None,
    pass_rule: str = "majority",
    min_voters: int = 2,
    on_result: Callable[[PairResult], None] | None = None,
) -> list[PairResult]:
    """Score a whole dataset. `on_result` fires per pair so callers can stream to disk."""
    results: list[PairResult] = []
    for i, pair in enumerate(pairs):
        pid = str(pair.get("id", i))
        verdicts = judge_pair(pair, judges)
        result = decide(pid, verdicts, arbiter=arbiter, pair=pair,
                        pass_rule=pass_rule, min_voters=min_voters)
        results.append(result)
        if on_result:
            on_result(result)
    return results

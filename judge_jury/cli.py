"""Command line: run the jury, or print a report from an existing run.

  judge-jury run pairs.jsonl --judges judges.yaml --out verdicts.jsonl
  judge-jury demo                       # offline, mock judges, no network
  judge-jury report verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from judge_jury.backends import Judge
from judge_jury.jury import PairResult, run_jury
from judge_jury.metrics import outcome_tally, arbiter_tally, judge_stats
from judge_jury.reasons import cluster_fail_reasons


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_judges(path: str) -> tuple[list[Judge], Judge | None, str, int]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    judges = [Judge(**j) for j in cfg["judges"]]
    arbiter = None
    if cfg.get("arbiter"):
        by_name = {j.name: j for j in judges}
        arbiter = by_name.get(cfg["arbiter"])
        if arbiter is None:
            raise SystemExit(f"arbiter '{cfg['arbiter']}' is not one of the judges")
    return judges, arbiter, cfg.get("pass_rule", "majority"), int(cfg.get("min_voters", 2))


FINISHED = ("pass", "fail", "conflict")


def _finished_rows(path: str) -> list[dict[str, Any]]:
    """Rows of an earlier run that reached a verdict. Pending rows (judge errors) are judged again."""
    if not Path(path).exists():
        return []
    return [row for row in _read_jsonl(path) if row.get("outcome") in FINISHED]


def _cmd_run(args: argparse.Namespace) -> int:
    pairs = _read_jsonl(args.pairs)
    judges, arbiter, pass_rule, min_voters = _load_judges(args.judges)

    kept: list[dict[str, Any]] = []
    if args.out and not args.fresh:
        kept = _finished_rows(args.out)
        done = {str(row.get("pair_id")) for row in kept}
        todo = [p for i, p in enumerate(pairs) if str(p.get("id", i)) not in done]
        if kept:
            print(f"resuming: {len(kept)} pairs already judged in {args.out}, "
                  f"{len(todo)} left (use --fresh to start over)", file=sys.stderr)
        pairs = todo

    out = None
    if args.out:
        out = open(args.out, "w", encoding="utf-8")
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

    def sink(r: PairResult) -> None:
        print(f"  {r.pair_id:>8}  {r.outcome:<9} ({r.n_pass}p/{r.n_fail}f/{r.n_error}e)", file=sys.stderr)
        if out:
            out.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            out.flush()

    print(f"judging {len(pairs)} pairs with {len(judges)} judges"
          f"{' + arbiter' if arbiter else ''}...", file=sys.stderr)
    results = run_jury(pairs, judges, arbiter=arbiter, pass_rule=pass_rule,
                       min_voters=min_voters, on_result=sink)
    if out:
        out.close()
    _print_report([_result_from_dict(row) for row in kept] + results)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from judge_jury.mock import demo_jury

    demo_path = Path(__file__).resolve().parent.parent / "examples" / "code_review_pairs.jsonl"
    path = args.pairs or str(demo_path)
    pairs = _read_jsonl(path)
    judges, arbiter = demo_jury()
    print(f"[demo] offline mock judges, no network. {len(pairs)} pairs from {path}\n", file=sys.stderr)
    results = run_jury(pairs, judges, arbiter=arbiter)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    _print_report(results)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.verdicts)
    results = [_result_from_dict(r) for r in rows]
    _print_report(results)
    return 0


def _result_from_dict(d: dict[str, Any]) -> PairResult:
    from judge_jury.protocol import Verdict

    r = PairResult(pair_id=str(d.get("pair_id", "")), outcome=d.get("outcome", "pending"))
    r.n_pass, r.n_fail, r.n_error = d.get("n_pass", 0), d.get("n_fail", 0), d.get("n_error", 0)
    r.verdicts = [Verdict(**v) for v in d.get("verdicts", [])]
    if d.get("arbiter"):
        r.arbiter = Verdict(**d["arbiter"])
    return r


def _print_report(results: list[PairResult]) -> None:
    tally = outcome_tally(results)
    n_judges = max((len(r.verdicts) for r in results), default=0)
    has_arb = any(r.arbiter for r in results)

    print(f"\n{len(results)} pairs · {n_judges} judges{' + arbiter' if has_arb else ''}")
    for k in ("pass", "fail", "conflict", "pending"):
        line = f"  {k:<9} {tally.get(k, 0):>5}"
        if k == "conflict" and has_arb:
            arb = arbiter_tally(results)
            line += f"  -> arbiter: {arb['pass']} pass / {arb['fail']} fail"
        print(line)
    n_error_pairs = sum(1 for r in results if r.n_error and r.outcome == "pending")
    if n_error_pairs:
        print(f"  {'(pending from judge errors, re-run these)':<40} {n_error_pairs}")

    reasons = cluster_fail_reasons(results)
    if reasons:
        print("\ntop fail reasons")
        for label, n in reasons:
            print(f"  {n:>4}  {label}")

    stats = judge_stats(results)
    if stats:
        print("\ninter-judge agreement")
        differ = len(stats) > 1 and stats[0].fail_rate > stats[-1].fail_rate
        strict = stats[0].name if differ else ""
        soft = stats[-1].name if differ else ""
        for s in stats:
            tag = "strictest" if s.name == strict else ("softest" if s.name == soft else "")
            print(f"  {s.name:<14} {tag:<10} (fail rate {s.fail_rate:.2f}, {s.errored} errors)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judge-jury", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="judge a JSONL file of pairs")
    r.add_argument("pairs")
    r.add_argument("--judges", required=True, help="judges.yaml")
    r.add_argument("--out", help="write per-pair verdicts here (JSONL); an existing file is resumed")
    r.add_argument("--fresh", action="store_true", help="ignore an existing --out file and judge everything again")
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("demo", help="run offline with mock judges (no network)")
    d.add_argument("pairs", nargs="?", help="optional JSONL, defaults to the bundled example")
    d.add_argument("--out", help="write per-pair verdicts here (JSONL)")
    d.set_defaults(func=_cmd_demo)

    rep = sub.add_parser("report", help="print a report from a verdicts file")
    rep.add_argument("verdicts")
    rep.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

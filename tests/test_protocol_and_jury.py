"""Offline tests. No network: the mock judges make the jury runnable in CI."""

from judge_jury.protocol import parse_verdict, Verdict
from judge_jury.jury import decide, run_jury
from judge_jury.mock import demo_jury, MockJudge


def test_parse_clean_json():
    v = parse_verdict('{"verdict":"pass","confidence":0.9,"fatal":false,"scores":{"correctness":3,"completeness":3,"grounding":4},"reasons":["ok"]}', judge="j")
    assert v.ok and v.verdict == "pass"
    assert v.scores["grounding"] == 4
    assert v.judge == "j"


def test_parse_fenced_and_preamble():
    text = "Here is my verdict:\n```json\n{\"verdict\": \"fail\", \"fatal\": true, \"reasons\": [\"nope\"]}\n```\nDone."
    v = parse_verdict(text)
    assert v.ok and v.verdict == "fail" and v.fatal is True


def test_parse_garbage_becomes_error():
    v = parse_verdict("I cannot answer that.")
    assert not v.ok and v.error is not None


def test_fatal_overrides_majority():
    verdicts = [
        Verdict(verdict="pass", judge="a"),
        Verdict(verdict="pass", judge="b"),
        Verdict(verdict="fail", fatal=True, judge="c"),
    ]
    r = decide("p1", verdicts)
    assert r.outcome == "fail"      # one fatal beats two soft passes


def test_pending_when_too_few_voters():
    verdicts = [
        Verdict(verdict="", judge="a", error="timeout"),
        Verdict(verdict="pass", judge="b"),
    ]
    r = decide("p2", verdicts, min_voters=2)
    assert r.outcome == "pending"   # only one real vote


def test_conflict_goes_to_arbiter():
    verdicts = [Verdict(verdict="pass", judge="a"), Verdict(verdict="fail", judge="b")]
    arbiter = MockJudge("arb")
    r = decide("p3", verdicts, arbiter=arbiter, pair={"instruction": "x", "response": "def f(): return 1"})
    assert r.outcome in ("pass", "fail")   # arbiter broke the tie
    assert r.arbiter is not None


def test_full_run_on_examples():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "examples" / "code_review_pairs.jsonl"
    pairs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    judges, arbiter = demo_jury()
    results = run_jury(pairs, judges, arbiter=arbiter)
    assert len(results) == len(pairs)
    outcomes = {r.outcome for r in results}
    assert outcomes <= {"pass", "fail", "conflict", "pending"}
    # The bundled set is built to produce both passes and fails.
    assert any(r.outcome == "pass" for r in results)
    assert any(r.outcome == "fail" for r in results)


def test_run_resumes_from_existing_out(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from judge_jury import cli

    monkeypatch.setattr(cli, "_load_judges", lambda path: (*demo_jury(), "majority", 2))
    seen = []
    real_run = cli.run_jury
    monkeypatch.setattr(cli, "run_jury", lambda pairs, *a, **k: seen.append(len(pairs)) or real_run(pairs, *a, **k))

    pairs = Path(__file__).resolve().parent.parent / "examples" / "code_review_pairs.jsonl"
    out = tmp_path / "verdicts.jsonl"
    cli.main(["run", str(pairs), "--judges", "x.yaml", "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    # Simulate a crash after 5 pairs: the second run only judges the rest.
    out.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    cli.main(["run", str(pairs), "--judges", "x.yaml", "--out", str(out)])
    assert seen == [total, total - 5]
    ids = [json.loads(l)["pair_id"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == total and len(set(ids)) == total

    # --fresh ignores the file and judges everything again.
    cli.main(["run", str(pairs), "--judges", "x.yaml", "--out", str(out), "--fresh"])
    assert seen[-1] == total


def test_report_on_real_run(capsys):
    from pathlib import Path
    from judge_jury.cli import main

    verdicts = Path(__file__).resolve().parent.parent / "examples" / "real_run_verdicts.jsonl"
    assert main(["report", str(verdicts)]) == 0
    out = capsys.readouterr().out
    assert "16 pairs · 4 judges" in out
    assert "pass         10" in out and "fail          6" in out
    # all four judges have the same fail rate, so none is labelled strictest/softest
    assert "strictest" not in out and "softest" not in out

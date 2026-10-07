# llm-judge-jury

Score instruction/response pairs with a jury of independent LLM judges, then keep only the pairs a majority agrees are good.

If you build SFT or preference datasets, you already know the problem. A synthetic pair looks fine, passes a glance, and only later turns out to be subtly wrong: the code doesn't compile, the answer drifts from the source, a flag is invented. One judge model has the same blind spots as your generator. A jury of different models, voting independently, catches what any single reviewer misses.

This is a small, backend-agnostic tool for exactly that. Point it at your pairs, list a few judges (local or hosted, anything OpenAI-compatible), and it returns a per-pair consensus verdict with the reasons attached.

## What it does

- Sends each pair to several judges with a strict verdict schema. No prose, just a JSON object per judge.
- Takes a majority vote. A pair passes only when enough judges independently agree.
- Sends conflicts to an arbiter judge instead of coin-flipping a tie.
- Never dies on one bad judge. A timeout or a refusal becomes an `error` record, the pair stays `pending`, the run continues.
- Clusters the fail reasons so you can see what your generator gets wrong most often.
- Reports inter-judge agreement, so you learn which judge is strict and which rubber-stamps.

## The verdict schema

Every judge returns the same object. Strict, because free-text reviews can't be counted.

```json
{
  "verdict": "pass",
  "confidence": 0.82,
  "fatal": false,
  "scores": { "correctness": 3, "completeness": 4, "grounding": 4 },
  "reasons": ["compiles and handles the empty-list case"],
  "needs_check": []
}
```

`fatal` is separate from the scores on purpose. A pair can score decently and still be fatal (wrong API, made-up function), and a fatal flag from any judge forces the pair out regardless of the vote.

## Quick start

```bash
pip install -e .
```

You need at least one OpenAI-compatible endpoint. A local Ollama or llama.cpp server works, so does a hosted API. Judges are declared in a small config:

```yaml
# judges.yaml
judges:
  - name: local-a
    base_url: http://localhost:11434/v1
    model: qwen2.5:32b
  - name: local-b
    base_url: http://localhost:8080/v1
    model: llama-3.3-70b
  - name: local-c
    base_url: http://localhost:1234/v1
    model: mistral-small
arbiter: local-a          # breaks conflicts
pass_rule: majority       # or "unanimous", or "k=2"
```

Run the jury over a JSONL file of pairs:

```bash
judge-jury run examples/code_review_pairs.jsonl --judges judges.yaml --out verdicts.jsonl
```

If the run stops halfway (a crash, a closed laptop, a judge stuck on rate limits), run the same command again. Pairs that already got a verdict in `verdicts.jsonl` are kept and skipped; pairs left `pending` by judge errors are judged again. Add `--fresh` to ignore the file and start over.

Then look at what happened:

```bash
judge-jury report verdicts.jsonl
```

```
1000 pairs · 3 judges + arbiter
  pass       684
  fail       229
  conflict    71  → arbiter: 44 pass / 27 fail
  error        16  (judge timeouts, re-run these)

top fail reasons
  128  does not compile / runtime error
   61  drifts from the reference
   40  missing edge case
   22  invented API or flag

inter-judge agreement
  local-a  strictest  (fail rate 0.28)
  local-b             (fail rate 0.19)
  local-c  softest    (fail rate 0.11)
```

## Input format

One JSON object per line. `instruction` and `response` are required. Anything under `context` is passed to the judges as grounding they must check the response against.

```json
{"id": "cr-014", "instruction": "Review this function for correctness.", "response": "...", "context": {"language": "python", "reference": "..."}}
```

The `examples/` folder has a small code-review set you can run immediately to see the shape of the output.

## Why a jury and not one strong judge

A single judge is one distribution. It is confidently wrong in its own consistent way, and if it shares training lineage with your generator it is wrong in the *same* way, so it waves through exactly the pairs you most need caught. Independent judges fail differently. The pairs all of them pass are the ones you can trust. The pairs they split on are the interesting ones, and those go to the arbiter rather than to a coin.

This does cost N calls per pair. For a dataset you are going to train on for weeks, catching a few hundred bad pairs up front is worth it. For a throwaway set, use one judge.

## Design notes

- **Judges never see each other's votes.** Independence is the whole point, so each call is isolated.
- **A refusal is data, not a crash.** Some judges decline some inputs. That becomes an `error` record and the pair stays pending for a re-run or a different judge, rather than silently dropping.
- **Scores are advisory, the vote decides.** The numeric axes help you cluster and debug. Pass/fail is the majority of the `verdict` field, with `fatal` as an override.
- **Rate limits are expected.** Each backend retries with backoff and the run is resumable, so a mid-run 429 doesn't lose the pairs already judged.

## Limitations

- Judges share some biases no matter how you pick them. A jury narrows the blind spot, it doesn't remove it. Spot-check a sample by hand.
- Consensus measures agreement, not truth. If every judge is wrong the same way, the pair passes. Diverse judges matter more than many judges.
- The verdict schema is opinionated. Swap the axes in `protocol.py` for your task.

## License

MIT.

Built while assembling training data for [CastelOS](https://github.com/CastelDazur/castelos-public). The pipeline is generic; the examples here are code-review pairs.

# Contributing

Small, focused PRs are easiest to merge. If you're not sure something fits, open an issue first.

## Running it

```bash
pip install -e .
pip install pytest
pytest -q
judge-jury demo      # offline, no network, uses the bundled example pairs
```

The demo uses mock judges (`judge_jury/mock.py`) so everything runs without a model server. Real judges are declared in a YAML file, see `judges.example.yaml`.

## Good first issues

- Add a judge backend for an API that isn't plain OpenAI-compatible.
- Add reason buckets in `reasons.py` for a domain other than code review.
- A `--resume` flag for `run` that skips pairs already in the output file.
- Cohen's kappa alongside the raw agreement in `metrics.py`.

## Style

- Standard library where it's enough. The only runtime deps are `requests` and `pyyaml`.
- Keep judges independent. Nothing in the jury should let one judge see another's vote.
- A judge failing is normal. Handle it as an `error` verdict, never a crash.

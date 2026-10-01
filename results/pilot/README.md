# Pilot results

Checked-in snapshot of the constrained live pilot (not the full 500K experiment).

## run-df6b3a158403

| Field | Value |
|---|---|
| Provider | tinker |
| Model | `Qwen/Qwen3.5-9B` |
| Haystack (approx) | ~5,020 tokens |
| Prompt tokens (API) | 9,531 |
| Completion tokens | 2,778 |
| Finish reason | `stop` |
| Latency | ~14.6 s |
| Stub-judge retrieval | 1.0 |
| Stub-judge causal | 1.0 |

Files:

- `eval_capture.json` — run metadata (haystack body omitted from request dump)
- `model_raw_output.txt` — full trajectory (`reasoning_content` + answer)
- `judge_verdict.json` — stub judge output
- `haystack.manifest.json` — clue placements for that run’s haystack

Regenerate matching haystacks with `seed=42` and `--target-tokens 5000`.

# llm-chaos-experiment

Telemetry evaluation harness for testing **open-weights reasoning models** on:

1. **Long-context retrieval** — sparse anomaly clues buried in large mock system logs
2. **Multi-step causal trajectories** — reconstructing an incident chain (A → B → C)

## Original experiment design

| Piece | Intent |
|---|---|
| `haystack_generator.py` | Build a large mock telemetry corpus (target ~500K tokens) and inject chronological anomaly clues **A, B, C** at explicit depth percentiles (e.g. 10% / 50% / 90%) |
| `eval_pipeline.py` | Load/generate the haystack, call an external LLM (Anthropic / Together / Tinker), capture the raw reasoning trajectory |
| `judge.py` | Score the trajectory against a golden rubric on **Retrieval Success** and **Causal Logic** (LLM-as-a-judge) |

**Ground-truth causal story**

- **A** — auth token issuer clock skew / missed key rotation  
- **B** — API gateway JWT rejection spike (downstream of A)  
- **C** — billing pipeline cascade abort correlated with auth failures  

A correct trajectory must **retrieve** all three clues and **not invert** causality.

## What is implemented today (constrained pilot)

| Item | Status |
|---|---|
| Haystack generator + depth placement + manifests | Done |
| Eval pipeline + stub provider | Done |
| Tinker provider (OpenAI-compatible HTTP) | Done |
| Anthropic / Together clients | Placeholders only |
| Judge rubric + prompt schema | Done |
| Real LLM-as-a-judge | **Not done** — runtime path used is a stub heuristic |
| Live model @ ~500K context | **Not done** |
| Live pilot @ ~5K context | Done — see below |

### Pilot result (claim carefully)

- **Run:** `results/pilot/run-df6b3a158403/`
- **Provider / model:** Tinker / `Qwen/Qwen3.5-9B`
- **Haystack size:** ~5K tokens (not 500K)
- **Outcome:** model cited clues A/B/C and stated A→B→C; `finish_reason=stop`; stub-judge scores 1.0 / 1.0
- **Meaning:** shows short-context causal reconstruction when needles are easy to see. It does **not** demonstrate long-context needle retrieval under depth stress.

A full ~500K haystack was generated locally during development (`artifacts/haystack.txt`) but is **not** committed (regenerate with the generator). Stub-only runs at 500K only prove the pipeline, not model capability.

### Known constraints

1. **Context window** — `Qwen/Qwen3.5-9B` on Tinker is ~64K; it cannot ingest the designed 500K haystack as-is.
2. **Python** — project was developed on **3.9**. Official `tinker` SDK wants **≥3.11**; the Tinker client uses stdlib HTTP instead.
3. **Cloudflare** — Tinker HTTP requires a non-default `User-Agent` (already set in `TinkerLLMClient`).
4. **Thinking tokens** — hybrid models fill `reasoning_content` first; low `max_tokens` truncates mid-trace. Pilot used 4096.
5. **n=1** — no seed/depth/model matrix yet.
6. **Judge** — not a real LLM judge yet.

## Repo layout

```
haystack_generator.py   # corpus + clue injection
eval_pipeline.py        # provider calls + capture
judge.py                # rubric + grading
results/pilot/          # checked-in pilot capture (no large haystacks)
requirements.txt
.gitignore
README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # currently empty / optional; stdlib is enough for stub+Tinker HTTP

# For live Tinker runs:
export TINKER_API_KEY='...'
```

## Quickstart

```bash
# 1) Generate a small haystack (pilot-scale)
python haystack_generator.py --target-tokens 5000 --depth-a 10 --depth-b 50 --depth-c 90

# 2a) Stub end-to-end (no API key)
python eval_pipeline.py --provider stub --target-tokens 5000

# 2b) Live Tinker run (reuse a generated file to avoid regen)
python eval_pipeline.py \
  --provider tinker \
  --model Qwen/Qwen3.5-9B \
  --haystack-path artifacts/haystack.txt \
  --target-tokens 5000

# 3) Grade capture (stub judge today)
python judge.py --capture artifacts/<run_id>/eval_capture.json
```

Programmatic Tinker run with higher `max_tokens` (CLI does not expose it yet):

```python
from pathlib import Path
from haystack_generator import HaystackConfig, clues_from_mapping
from eval_pipeline import EvalRunConfig, IncidentTaskConfig, run_eval

cfg = EvalRunConfig(
    haystack=HaystackConfig(
        target_tokens=5000,
        seed=42,
        clues=clues_from_mapping({"A": 10, "B": 50, "C": 90}),
    ),
    task=IncidentTaskConfig(
        provider="tinker",
        model="Qwen/Qwen3.5-9B",
        max_tokens=4096,
        temperature=0.0,
    ),
    artifacts_dir=Path("artifacts"),
)
print(run_eval(cfg).artifacts)
```

## Next steps (to land the original experiment)

1. **Pick a context-capable path** — use an extended-context Tinker checkpoint, or redefine the long-context target to 32K/64K and treat 500K as aspirational.
2. **Run a matrix** — same haystack SHA(s) × depths × context sizes × ≥2 open-weights models × multiple seeds; report pass rates.
3. **Implement real LLM-as-a-judge** — wire `AnthropicJudgeClient` (or Tinker) to the existing rubric prompt; store raw judge JSON; spot-check agreement.
4. **Clarify “agent trajectory”** — either keep single-shot CoT and rename the claim, or add a real tool loop over logs.
5. **Harden CLI** — expose `--max-tokens`; pin model IDs; fail closed on `finish_reason=length`; prefer Python ≥3.11 if adopting the official Tinker SDK.
6. **Pre-register success criteria** — retrieval = all three clue IDs with evidence; causal = order A→B→C and no inverted root cause.

## Collaboration notes

- Do **not** commit API keys or `.env` files.
- Large `artifacts/` outputs are gitignored; keep small pilot summaries under `results/`.
- Prefer private repo + explicit collaborator invite for shared Tinker work.

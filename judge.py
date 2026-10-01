"""
LLM-as-a-judge for incident-response trajectories.

Grades a model's raw reasoning trace against an explicit golden rubric on:
  1) Retrieval Success — did it surface clues A/B/C?
  2) Causal Logic — did it order and link A → B → C correctly?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Placeholder schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RubricClue:
    """Expected evidence the subject model should retrieve."""

    clue_id: str
    must_include_any: Tuple[str, ...]  # substrings / markers that count as a hit
    summary: str


@dataclass
class GoldenRubric:
    """
    Explicit scoring contract for a scenario.

    retrieval_success: coverage of planted clues.
    causal_logic: correct temporal/causal chain A → B → C (and no inverted cause).
    """

    rubric_id: str
    scenario_summary: str
    clues: Tuple[RubricClue, ...]
    required_causal_order: Tuple[str, ...]  # e.g. ("A", "B", "C")
    causal_narrative_keypoints: Tuple[str, ...]
    # Weights in [0, 1]; should typically sum to 1.0.
    weight_retrieval: float = 0.5
    weight_causal: float = 0.5
    pass_threshold: float = 0.7

    def __post_init__(self) -> None:
        total = self.weight_retrieval + self.weight_causal
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Rubric weights must sum to 1.0, got {total}")


@dataclass
class DimensionScore:
    """Score for one grading dimension."""

    name: str
    score: float  # 0.0–1.0
    rationale: str
    evidence_spans: List[str] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)


@dataclass
class JudgeVerdict:
    """Final structured judgment."""

    rubric_id: str
    run_id: str
    retrieval_success: DimensionScore
    causal_logic: DimensionScore
    overall_score: float
    passed: bool
    judge_model: str
    judge_provider: str
    judge_raw_text: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeRequest:
    """Inputs to the judge."""

    run_id: str
    subject_raw_text: str
    rubric: GoldenRubric
    # Optional: haystack placements / clue texts to ground the judge.
    clue_context: Dict[str, str] = field(default_factory=dict)
    judge_model: str = "placeholder-judge-model"
    judge_provider: str = "stub"
    temperature: float = 0.0
    max_tokens: int = 2048


# ---------------------------------------------------------------------------
# Default golden rubric (matches haystack_generator.DEFAULT_CLUES)
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = GoldenRubric(
    rubric_id="incident-cascade-v1",
    scenario_summary=(
        "Auth token issuer clock skew / missed key rotation (A) causes "
        "API gateway JWT rejection spike (B), which cascades into billing "
        "pipeline abort (C)."
    ),
    clues=(
        RubricClue(
            clue_id="A",
            must_include_any=("ANOMALY_CLUE_A", "clock_skew", "signing_key_id=legacy-2024"),
            summary="Auth issuer clock skew / missed key rotation",
        ),
        RubricClue(
            clue_id="B",
            must_include_any=("ANOMALY_CLUE_B", "jwt_reject_rate", "signature_invalid"),
            summary="API gateway JWT rejection spike tied to auth issuer",
        ),
        RubricClue(
            clue_id="C",
            must_include_any=("ANOMALY_CLUE_C", "billing.charge_pipeline", "INC-88421"),
            summary="Billing cascade abort correlated with auth failures",
        ),
    ),
    required_causal_order=("A", "B", "C"),
    causal_narrative_keypoints=(
        "Root cause originates in auth token issuer (clue A)",
        "Gateway JWT failures are downstream of A (clue B)",
        "Billing abort is a cascade from auth/gateway failures (clue C)",
        "Order must not claim billing caused auth skew",
    ),
)


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluation judge for long-context incident-response agents.
Score ONLY against the provided golden rubric. Do not reward fluency alone.
Return valid JSON matching the schema described by the user.
"""

JUDGE_USER_PROMPT_TEMPLATE = """\
## Golden rubric
rubric_id: {rubric_id}
scenario: {scenario_summary}
required_causal_order: {required_causal_order}
clues:
{clues_block}
causal_narrative_keypoints:
{keypoints_block}

## Optional planted clue text (ground truth excerpts)
{clue_context_block}

## Subject model trajectory (raw text)
```
{subject_raw_text}
```

## Scoring dimensions
1. retrieval_success (0.0–1.0): fraction of clues A/B/C correctly surfaced,
   with brief evidence spans copied from the subject text.
2. causal_logic (0.0–1.0): whether the subject links clues in the required
   order and matches the causal keypoints (penalize inverted causality).

## Output JSON schema (strict)
{{
  "retrieval_success": {{
    "score": <float 0..1>,
    "rationale": <string>,
    "evidence_spans": [<string>, ...],
    "missing_items": [<clue_id or description>, ...]
  }},
  "causal_logic": {{
    "score": <float 0..1>,
    "rationale": <string>,
    "evidence_spans": [<string>, ...],
    "missing_items": [<string>, ...]
  }},
  "overall_notes": <string>
}}
"""


def render_judge_prompt(req: JudgeRequest) -> Tuple[str, str]:
    clues_block = "\n".join(
        f"- {c.clue_id}: {c.summary} | markers={list(c.must_include_any)}"
        for c in req.rubric.clues
    )
    keypoints_block = "\n".join(f"- {k}" for k in req.rubric.causal_narrative_keypoints)
    if req.clue_context:
        clue_context_block = "\n".join(f"{k}: {v}" for k, v in req.clue_context.items())
    else:
        clue_context_block = "(none provided)"

    user = JUDGE_USER_PROMPT_TEMPLATE.format(
        rubric_id=req.rubric.rubric_id,
        scenario_summary=req.rubric.scenario_summary,
        required_causal_order=list(req.rubric.required_causal_order),
        clues_block=clues_block,
        keypoints_block=keypoints_block,
        clue_context_block=clue_context_block,
        subject_raw_text=req.subject_raw_text,
    )
    return JUDGE_SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Judge LLM interface (placeholder) + heuristic fallback
# ---------------------------------------------------------------------------

class JudgeClient(ABC):
    @abstractmethod
    def judge(self, system_prompt: str, user_prompt: str, model: str) -> Tuple[str, float]:
        """Return (raw_text, latency_ms)."""
        raise NotImplementedError


class StubJudgeClient(JudgeClient):
    """
    Offline heuristic judge used until a real LLM judge is wired.

    Not a substitute for LLM-as-judge quality — only for pipeline smoke tests.
    """

    def judge(self, system_prompt: str, user_prompt: str, model: str) -> Tuple[str, float]:
        t0 = time.perf_counter()
        # Pull subject text back out of the rendered prompt for heuristics.
        match = re.search(r"## Subject model trajectory.*?\n```\n(.*?)\n```", user_prompt, re.S)
        subject = match.group(1) if match else user_prompt
        subject_l = subject.lower()

        retrieved = []
        missing = []
        for clue_id, markers in _markers_from_prompt_fallback():
            hit = any(m.lower() in subject_l for m in markers)
            (retrieved if hit else missing).append(clue_id)

        retrieval_score = len(retrieved) / max(1, len(retrieved) + len(missing))

        # Crude causal check: look for A...B...C order in text.
        positions = []
        for cid in ("A", "B", "C"):
            idx = subject_l.find(f"clue_{cid.lower()}") 
            if idx < 0:
                idx = subject_l.find(f"anomaly_clue_{cid.lower()}")
            positions.append(idx)

        if all(p >= 0 for p in positions) and positions == sorted(positions):
            causal_score = 1.0
            causal_missing: List[str] = []
            causal_rationale = "Markers appear in A→B→C order."
        elif sum(1 for p in positions if p >= 0) >= 2:
            causal_score = 0.4
            causal_missing = ["strict A→B→C ordering"]
            causal_rationale = "Partial clue mentions without clear ordered causal chain."
        else:
            causal_score = 0.0
            causal_missing = ["causal chain A→B→C"]
            causal_rationale = "Insufficient evidence of causal linking."

        payload = {
            "retrieval_success": {
                "score": retrieval_score,
                "rationale": f"Heuristic marker hits: {retrieved}",
                "evidence_spans": retrieved,
                "missing_items": missing,
            },
            "causal_logic": {
                "score": causal_score,
                "rationale": causal_rationale,
                "evidence_spans": [],
                "missing_items": causal_missing,
            },
            "overall_notes": "Produced by StubJudgeClient heuristic — replace with LLM judge.",
        }
        latency = (time.perf_counter() - t0) * 1000.0
        return json.dumps(payload, indent=2), latency


def _markers_from_prompt_fallback() -> List[Tuple[str, Tuple[str, ...]]]:
    return [
        ("A", ("ANOMALY_CLUE_A", "clock_skew", "signing_key_id=legacy-2024")),
        ("B", ("ANOMALY_CLUE_B", "jwt_reject_rate", "signature_invalid")),
        ("C", ("ANOMALY_CLUE_C", "billing.charge_pipeline", "INC-88421")),
    ]


class AnthropicJudgeClient(JudgeClient):
    """Placeholder LLM-as-judge via Anthropic."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def judge(self, system_prompt: str, user_prompt: str, model: str) -> Tuple[str, float]:
        raise NotImplementedError(
            "AnthropicJudgeClient is a placeholder. "
            "Call Messages API and return the assistant text + latency."
        )


def build_judge_client(provider: str) -> JudgeClient:
    mapping = {
        "stub": StubJudgeClient,
        "anthropic": AnthropicJudgeClient,
    }
    if provider not in mapping:
        raise ValueError(f"Unknown judge provider '{provider}'. Choose from: {sorted(mapping)}")
    return mapping[provider]()


# ---------------------------------------------------------------------------
# Parsing + scoring
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from judge raw text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Judge output did not contain a JSON object.")
    return json.loads(match.group(0))


def _dimension_from_payload(name: str, blob: Dict[str, Any]) -> DimensionScore:
    return DimensionScore(
        name=name,
        score=float(blob.get("score", 0.0)),
        rationale=str(blob.get("rationale", "")),
        evidence_spans=list(blob.get("evidence_spans") or []),
        missing_items=list(blob.get("missing_items") or []),
    )


def grade_trajectory(
    req: JudgeRequest,
    client: Optional[JudgeClient] = None,
) -> JudgeVerdict:
    """
    Run LLM-as-a-judge (or stub) and produce a structured JudgeVerdict.
    """
    system_prompt, user_prompt = render_judge_prompt(req)
    judge = client or build_judge_client(req.judge_provider)
    raw_text, latency_ms = judge.judge(system_prompt, user_prompt, req.judge_model)

    payload = _extract_json_object(raw_text)
    retrieval = _dimension_from_payload("retrieval_success", payload.get("retrieval_success", {}))
    causal = _dimension_from_payload("causal_logic", payload.get("causal_logic", {}))

    overall = (
        req.rubric.weight_retrieval * retrieval.score
        + req.rubric.weight_causal * causal.score
    )
    return JudgeVerdict(
        rubric_id=req.rubric.rubric_id,
        run_id=req.run_id,
        retrieval_success=retrieval,
        causal_logic=causal,
        overall_score=overall,
        passed=overall >= req.rubric.pass_threshold,
        judge_model=req.judge_model,
        judge_provider=req.judge_provider,
        judge_raw_text=raw_text,
        latency_ms=latency_ms,
        metadata={"overall_notes": payload.get("overall_notes")},
    )


def grade_eval_capture(
    capture_path: Path,
    rubric: GoldenRubric = DEFAULT_RUBRIC,
    judge_provider: str = "stub",
    judge_model: str = "placeholder-judge-model",
    out_path: Optional[Path] = None,
) -> JudgeVerdict:
    """Load an EvalCapture JSON from eval_pipeline and grade it."""
    data = json.loads(Path(capture_path).read_text(encoding="utf-8"))
    clue_context: Dict[str, str] = {}
    # Prefer haystack manifest beside the capture if present.
    artifacts = data.get("artifacts") or {}
    haystack_path = artifacts.get("haystack")
    if haystack_path:
        manifest = Path(haystack_path).with_suffix(Path(haystack_path).suffix + ".manifest.json")
        if manifest.exists():
            man = json.loads(manifest.read_text(encoding="utf-8"))
            for p in man.get("placements", []):
                clue_context[p["clue_id"]] = p.get("text", "")

    req = JudgeRequest(
        run_id=data.get("run_id", "unknown"),
        subject_raw_text=data.get("raw_text", ""),
        rubric=rubric,
        clue_context=clue_context,
        judge_model=judge_model,
        judge_provider=judge_provider,
    )
    verdict = grade_trajectory(req)

    dest = out_path or (Path(artifacts.get("dir", Path(capture_path).parent)) / "judge_verdict.json")
    Path(dest).write_text(json.dumps(asdict(verdict), indent=2), encoding="utf-8")
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade a model trajectory with LLM-as-a-judge.")
    p.add_argument("--capture", type=Path, required=True, help="Path to eval_capture.json")
    p.add_argument("--judge-provider", default="stub", choices=["stub", "anthropic"])
    p.add_argument("--judge-model", default="placeholder-judge-model")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    verdict = grade_eval_capture(
        capture_path=args.capture,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                "run_id": verdict.run_id,
                "passed": verdict.passed,
                "overall_score": verdict.overall_score,
                "retrieval_success": verdict.retrieval_success.score,
                "causal_logic": verdict.causal_logic.score,
                "missing_retrieval": verdict.retrieval_success.missing_items,
                "missing_causal": verdict.causal_logic.missing_items,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

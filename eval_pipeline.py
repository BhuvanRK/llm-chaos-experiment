"""
Evaluation pipeline: haystack → incident-response LLM call → raw capture.

Provider clients are intentionally stubbed so Anthropic / Together / Tinker
(or any OpenAI-compatible endpoint) can be plugged in behind a common interface.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from haystack_generator import (
    AnomalyClue,
    HaystackConfig,
    HaystackResult,
    clues_from_mapping,
    generate_haystack,
)


# ---------------------------------------------------------------------------
# Placeholder schemas
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """Normalized chat/completion request sent to a provider."""

    model: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = 4096
    temperature: float = 0.0
    # Extra provider-specific knobs (top_p, stop, thinking budget, etc.).
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized provider response."""

    raw_text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_provider_payload: Optional[Dict[str, Any]] = None


@dataclass
class IncidentTaskConfig:
    """How the model is asked to investigate the haystack."""

    task_id: str = "incident-cascade-v1"
    system_prompt: str = (
        "You are an on-call SRE performing incident response. "
        "Read the telemetry carefully. Identify anomaly signals, reconstruct "
        "the causal chain, and cite the evidence you used."
    )
    user_prompt_template: str = (
        "Incident ticket INC-88421: customers report failed checkouts.\n"
        "Below is a large telemetry dump. Find the root-cause chain and "
        "list supporting log lines in chronological order.\n\n"
        "--- TELEMETRY START ---\n{haystack}\n--- TELEMETRY END ---\n\n"
        "Respond with: (1) retrieved anomaly clues, (2) causal narrative, "
        "(3) recommended mitigation."
    )
    model: str = "placeholder-model"
    provider: str = "stub"  # stub | anthropic | together | tinker
    max_tokens: int = 4096
    temperature: float = 0.0
    api_base: Optional[str] = None
    api_key_env: str = "LLM_API_KEY"


@dataclass
class EvalRunConfig:
    """End-to-end run configuration."""

    haystack: HaystackConfig
    task: IncidentTaskConfig = field(default_factory=IncidentTaskConfig)
    artifacts_dir: Path = Path("artifacts")
    run_id: Optional[str] = None
    # If set, skip generation and load an existing haystack file.
    haystack_path: Optional[Path] = None


@dataclass
class EvalCapture:
    """Persisted raw model output + provenance for the judge stage."""

    run_id: str
    created_at: str
    task_id: str
    provider: str
    model: str
    raw_text: str
    latency_ms: float
    haystack_sha256: str
    haystack_approx_tokens: int
    clue_ids: List[str]
    request: Dict[str, Any]
    response_meta: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider interface (placeholders)
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Swap-in interface for Anthropic / Together / Tinker / etc."""

    name: str = "base"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Deterministic placeholder — no network calls."""

    name = "stub"

    def complete(self, request: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        # Pretend the model partially retrieves clues if they appear in-prompt.
        found = []
        for marker in ("ANOMALY_CLUE_A", "ANOMALY_CLUE_B", "ANOMALY_CLUE_C"):
            if marker in request.user_prompt:
                found.append(marker)
        body = (
            "[STUB MODEL OUTPUT]\n"
            f"Retrieved markers: {', '.join(found) or 'none'}\n"
            "Causal narrative: placeholder — wire a real provider client.\n"
            "Mitigation: placeholder.\n"
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            raw_text=body,
            model=request.model,
            provider=self.name,
            latency_ms=latency,
            input_tokens=None,
            output_tokens=None,
            finish_reason="stop",
            raw_provider_payload={"stub": True},
        )


class AnthropicLLMClient(LLMClient):
    """Placeholder for Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = api_base or "https://api.anthropic.com"

    def complete(self, request: LLMRequest) -> LLMResponse:
        # TODO: anthropic.Anthropic(...).messages.create(...)
        raise NotImplementedError(
            "AnthropicLLMClient.complete is a placeholder. "
            "Install `anthropic` and implement Messages API call."
        )


class TogetherLLMClient(LLMClient):
    """Placeholder for Together / OpenAI-compatible chat completions."""

    name = "together"

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.api_base = api_base or "https://api.together.xyz/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        # TODO: POST {api_base}/chat/completions
        raise NotImplementedError(
            "TogetherLLMClient.complete is a placeholder. "
            "Implement OpenAI-compatible chat.completions call."
        )


class TinkerLLMClient(LLMClient):
    """
    Tinker (Thinking Machines) via OpenAI-compatible chat completions.

    Uses stdlib HTTP so it works without the `tinker`/`openai` packages
    (those need Python >= 3.11; this project's venv is 3.9).

    Docs: https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
    Auth: TINKER_API_KEY
    """

    name = "tinker"
    DEFAULT_API_BASE = (
        "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    )

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        self.api_base = (api_base or self.DEFAULT_API_BASE).rstrip("/")

    def complete(self, request: LLMRequest) -> LLMResponse:
        import json as _json
        import urllib.error
        import urllib.request

        if not self.api_key:
            raise RuntimeError(
                "TINKER_API_KEY is not set. Export it or add it to ~/.zshrc, "
                "then ensure this shell has sourced it."
            )

        url = f"{self.api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.extra:
            payload.update(request.extra)

        body = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Cloudflare blocks Python-urllib's default UA (Error 1010).
                "User-Agent": "tinker-eval/0.1 (+https://thinkingmachines.ai/tinker)",
            },
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Tinker API HTTP {exc.code}: {err_body[:2000]}"
            ) from exc
        latency = (time.perf_counter() - t0) * 1000.0

        data = _json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or choice.get("text") or ""
        reasoning = message.get("reasoning_content") or ""
        # Hybrid/thinking models (e.g. Qwen3.5) often put CoT in reasoning_content.
        if reasoning and content:
            text = f"{reasoning}\n\n{content}"
        else:
            text = content or reasoning
        usage = data.get("usage") or {}

        return LLMResponse(
            raw_text=text,
            model=request.model,
            provider=self.name,
            latency_ms=latency,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw_provider_payload=data,
        )


def build_llm_client(provider: str, api_key: Optional[str] = None, api_base: Optional[str] = None) -> LLMClient:
    providers = {
        "stub": lambda: StubLLMClient(),
        "anthropic": lambda: AnthropicLLMClient(api_key=api_key, api_base=api_base),
        "together": lambda: TogetherLLMClient(api_key=api_key, api_base=api_base),
        "tinker": lambda: TinkerLLMClient(api_key=api_key, api_base=api_base),
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {sorted(providers)}")
    return providers[provider]()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_or_generate_haystack(config: EvalRunConfig) -> HaystackResult:
    if config.haystack_path:
        path = Path(config.haystack_path)
        text = path.read_text(encoding="utf-8")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Reconstruct a minimal HaystackResult for downstream capture.
        from haystack_generator import CHARS_PER_TOKEN_ESTIMATE, CluePlacement

        placements = [CluePlacement(**p) for p in manifest.get("placements", [])]
        approx_tokens = manifest.get("approx_token_count")
        if approx_tokens is None:
            approx_tokens = max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)
        return HaystackResult(
            text=text,
            config=config.haystack,
            placements=placements,
            approx_token_count=int(approx_tokens),
            char_count=len(text),
            line_count=text.count("\n") + (1 if text else 0),
            content_sha256=manifest.get("content_sha256", ""),
        )
    return generate_haystack(config.haystack)


def build_prompts(task: IncidentTaskConfig, haystack_text: str) -> LLMRequest:
    return LLMRequest(
        model=task.model,
        system_prompt=task.system_prompt,
        user_prompt=task.user_prompt_template.format(haystack=haystack_text),
        max_tokens=task.max_tokens,
        temperature=task.temperature,
    )


def run_eval(
    config: EvalRunConfig,
    client: Optional[LLMClient] = None,
) -> EvalCapture:
    """
    Generate/load haystack, call the LLM, and persist raw text + metadata.
    """
    run_id = config.run_id or f"run-{uuid.uuid4().hex[:12]}"
    artifacts_dir = Path(config.artifacts_dir) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    haystack = load_or_generate_haystack(config)
    haystack_path = artifacts_dir / "haystack.txt"
    haystack_path.write_text(haystack.text, encoding="utf-8")
    (artifacts_dir / "haystack.manifest.json").write_text(
        json.dumps(haystack.to_manifest(), indent=2), encoding="utf-8"
    )

    request = build_prompts(config.task, haystack.text)
    llm = client or build_llm_client(
        config.task.provider,
        api_key=os.getenv(config.task.api_key_env),
        api_base=config.task.api_base,
    )
    response = llm.complete(request)

    raw_path = artifacts_dir / "model_raw_output.txt"
    raw_path.write_text(response.raw_text, encoding="utf-8")

    capture = EvalCapture(
        run_id=run_id,
        created_at=_utc_now_iso(),
        task_id=config.task.task_id,
        provider=response.provider,
        model=response.model,
        raw_text=response.raw_text,
        latency_ms=response.latency_ms,
        haystack_sha256=haystack.content_sha256,
        haystack_approx_tokens=haystack.approx_token_count,
        clue_ids=[c.clue_id for c in haystack.config.clues] or [p.clue_id for p in haystack.placements],
        request={
            "model": request.model,
            "system_prompt": request.system_prompt,
            # Omit full haystack from JSON capture; kept on disk.
            "user_prompt_chars": len(request.user_prompt),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "extra": request.extra,
        },
        response_meta={
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "finish_reason": response.finish_reason,
            "raw_provider_payload": response.raw_provider_payload,
        },
        artifacts={
            "dir": str(artifacts_dir),
            "haystack": str(haystack_path),
            "raw_output": str(raw_path),
        },
    )

    capture_path = artifacts_dir / "eval_capture.json"
    # Persist without duplicating raw_text twice if desired — keep both for now.
    capture_path.write_text(json.dumps(asdict(capture), indent=2), encoding="utf-8")
    capture.artifacts["capture"] = str(capture_path)
    return capture


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run incident-response eval against a haystack.")
    p.add_argument("--target-tokens", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth-a", type=float, default=10.0)
    p.add_argument("--depth-b", type=float, default=50.0)
    p.add_argument("--depth-c", type=float, default=90.0)
    p.add_argument("--haystack-path", type=Path, default=None)
    p.add_argument("--provider", default="stub", choices=["stub", "anthropic", "together", "tinker"])
    p.add_argument("--model", default="placeholder-model")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--api-base", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    clues: Sequence[AnomalyClue] = clues_from_mapping(
        {"A": args.depth_a, "B": args.depth_b, "C": args.depth_c}
    )
    config = EvalRunConfig(
        haystack=HaystackConfig(
            target_tokens=args.target_tokens,
            seed=args.seed,
            clues=clues,
        ),
        task=IncidentTaskConfig(
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
        artifacts_dir=args.artifacts_dir,
        haystack_path=args.haystack_path,
    )
    capture = run_eval(config)
    print(
        json.dumps(
            {
                "run_id": capture.run_id,
                "provider": capture.provider,
                "model": capture.model,
                "latency_ms": capture.latency_ms,
                "haystack_approx_tokens": capture.haystack_approx_tokens,
                "artifacts": capture.artifacts,
                "raw_text_preview": capture.raw_text[:500],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

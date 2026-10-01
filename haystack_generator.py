"""
Haystack generator for long-context telemetry retrieval evals.

Builds a large mock system-log corpus (~500K tokens by default) and injects
chronological anomaly clues (A → B → C) at caller-specified depth percentiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Placeholder schemas
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN_ESTIMATE = 4  # rough English/log text heuristic


@dataclass(frozen=True)
class AnomalyClue:
    """A single chronological anomaly marker to bury in the haystack."""

    clue_id: str  # e.g. "A", "B", "C"
    text: str
    depth_percentile: float  # 0.0–100.0; e.g. 10.0 → ~10% into the corpus
    timestamp_iso: str  # chronological order should respect A < B < C

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth_percentile <= 100.0:
            raise ValueError(
                f"depth_percentile must be in [0, 100], got {self.depth_percentile}"
            )


@dataclass
class HaystackConfig:
    """Generation knobs for the mock telemetry corpus."""

    target_tokens: int = 500_000
    seed: int = 42
    line_prefix: str = "svc"
    services: Tuple[str, ...] = (
        "api-gateway",
        "auth",
        "billing",
        "cache",
        "worker",
        "db-proxy",
        "metrics",
        "ingress",
    )
    log_levels: Tuple[str, ...] = ("DEBUG", "INFO", "INFO", "INFO", "WARN")
    # Placeholder filler templates — replace with richer generators later.
    filler_templates: Tuple[str, ...] = (
        "request_id={rid} method=GET path=/v1/{svc}/health status=200 latency_ms={ms}",
        "request_id={rid} method=POST path=/v1/{svc}/events status=201 latency_ms={ms}",
        "cache_hit key={rid} ttl_s={ms} region=us-west-2",
        "db_query table={svc}_events rows={ms} duration_ms={ms}",
        "queue_depth topic={svc}.jobs depth={ms} consumers=4",
        "heartbeat component={svc} ok=true uptime_s={ms}",
    )
    clues: Sequence[AnomalyClue] = field(default_factory=tuple)
    output_path: Optional[Path] = None


@dataclass
class CluePlacement:
    """Where a clue landed after generation (for judge / analysis)."""

    clue_id: str
    depth_percentile_requested: float
    char_offset: int
    approx_token_offset: int
    line_index: int
    timestamp_iso: str
    text: str


@dataclass
class HaystackResult:
    """Generated corpus plus placement metadata."""

    text: str
    config: HaystackConfig
    placements: List[CluePlacement]
    approx_token_count: int
    char_count: int
    line_count: int
    content_sha256: str

    def to_manifest(self) -> Dict:
        """Serializable metadata (excludes full text body)."""
        return {
            "approx_token_count": self.approx_token_count,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "content_sha256": self.content_sha256,
            "target_tokens": self.config.target_tokens,
            "seed": self.config.seed,
            "placements": [asdict(p) for p in self.placements],
            "clues": [asdict(c) for c in self.config.clues],
        }


# ---------------------------------------------------------------------------
# Default chronological clues (placeholders)
# ---------------------------------------------------------------------------

DEFAULT_CLUES: Tuple[AnomalyClue, ...] = (
    AnomalyClue(
        clue_id="A",
        depth_percentile=10.0,
        timestamp_iso="2026-03-14T08:12:03Z",
        text=(
            "[ANOMALY_CLUE_A] auth.token_issuer clock_skew_ms=842 "
            "signing_key_id=legacy-2024 rotate_deadline_missed=true"
        ),
    ),
    AnomalyClue(
        clue_id="B",
        depth_percentile=50.0,
        timestamp_iso="2026-03-14T09:41:17Z",
        text=(
            "[ANOMALY_CLUE_B] api-gateway jwt_reject_rate=0.37 "
            "error=signature_invalid upstream=auth.token_issuer"
        ),
    ),
    AnomalyClue(
        clue_id="C",
        depth_percentile=90.0,
        timestamp_iso="2026-03-14T11:05:44Z",
        text=(
            "[ANOMALY_CLUE_C] billing.charge_pipeline cascade_abort "
            "reason=auth_failure_spike correlated_incident=INC-88421"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _estimate_chars_for_tokens(target_tokens: int) -> int:
    return target_tokens * CHARS_PER_TOKEN_ESTIMATE


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def _make_filler_line(rng: random.Random, cfg: HaystackConfig, seq: int) -> str:
    svc = rng.choice(cfg.services)
    level = rng.choice(cfg.log_levels)
    template = rng.choice(cfg.filler_templates)
    rid = hashlib.md5(f"{cfg.seed}:{seq}".encode()).hexdigest()[:12]
    ms = rng.randint(1, 900)
    # Synthetic monotonic-ish timestamp scaffolding (not wall-clock accurate).
    hh = (seq // 3600) % 24
    mm = (seq // 60) % 60
    ss = seq % 60
    body = template.format(rid=rid, svc=svc, ms=ms)
    return f"2026-03-14T{hh:02d}:{mm:02d}:{ss:02d}Z {level} {cfg.line_prefix}.{svc} {body}"


def _validate_clue_order(clues: Sequence[AnomalyClue]) -> None:
    if not clues:
        return
    by_depth = sorted(clues, key=lambda c: c.depth_percentile)
    by_time = sorted(clues, key=lambda c: c.timestamp_iso)
    if [c.clue_id for c in by_depth] != [c.clue_id for c in by_time]:
        raise ValueError(
            "Clue chronological order (timestamp_iso) must agree with "
            "depth_percentile order so A→B→C remains causal in-context."
        )


def generate_haystack(config: Optional[HaystackConfig] = None) -> HaystackResult:
    """
    Generate a large mock log string and inject clues at depth percentiles.

    Depth is measured over the *filler* character budget so clue size does not
    shift later percentiles unexpectedly.
    """
    cfg = config or HaystackConfig(clues=DEFAULT_CLUES)
    if not cfg.clues:
        cfg = HaystackConfig(
            target_tokens=cfg.target_tokens,
            seed=cfg.seed,
            line_prefix=cfg.line_prefix,
            services=cfg.services,
            log_levels=cfg.log_levels,
            filler_templates=cfg.filler_templates,
            clues=DEFAULT_CLUES,
            output_path=cfg.output_path,
        )

    _validate_clue_order(cfg.clues)
    rng = random.Random(cfg.seed)

    target_chars = _estimate_chars_for_tokens(cfg.target_tokens)
    clue_chars = sum(len(c.text) + 1 for c in cfg.clues)  # +1 newline
    filler_budget = max(0, target_chars - clue_chars)

    # Build filler lines until we cover the budget.
    lines: List[str] = []
    filled = 0
    seq = 0
    while filled < filler_budget:
        line = _make_filler_line(rng, cfg, seq)
        lines.append(line)
        filled += len(line) + 1
        seq += 1

    filler_text = "\n".join(lines)
    filler_len = len(filler_text)

    # Map percentile → character offset within filler, then splice clues.
    # Process from end → start so earlier offsets stay valid.
    sorted_clues = sorted(cfg.clues, key=lambda c: c.depth_percentile, reverse=True)
    pieces: List[str] = [filler_text]
    # Track (offset_in_final_approx) via incremental splice into a single buffer.
    buffer = filler_text
    raw_placements: List[Tuple[AnomalyClue, int]] = []

    for clue in sorted_clues:
        offset = int(filler_len * (clue.depth_percentile / 100.0))
        offset = min(max(0, offset), len(buffer))
        # Snap to nearest newline boundary for clean log lines.
        nl = buffer.rfind("\n", 0, offset)
        if nl != -1:
            offset = nl + 1
        injection = clue.text if clue.text.endswith("\n") else clue.text + "\n"
        buffer = buffer[:offset] + injection + buffer[offset:]
        raw_placements.append((clue, offset))

    # Recompute placements against the final buffer (offsets already absolute).
    placements: List[CluePlacement] = []
    for clue, offset in sorted(raw_placements, key=lambda x: x[1]):
        line_index = buffer.count("\n", 0, offset)
        placements.append(
            CluePlacement(
                clue_id=clue.clue_id,
                depth_percentile_requested=clue.depth_percentile,
                char_offset=offset,
                approx_token_offset=_estimate_tokens(buffer[:offset]),
                line_index=line_index,
                timestamp_iso=clue.timestamp_iso,
                text=clue.text,
            )
        )

    digest = hashlib.sha256(buffer.encode("utf-8")).hexdigest()
    result = HaystackResult(
        text=buffer,
        config=cfg,
        placements=placements,
        approx_token_count=_estimate_tokens(buffer),
        char_count=len(buffer),
        line_count=buffer.count("\n") + (1 if buffer else 0),
        content_sha256=digest,
    )

    if cfg.output_path:
        _write_haystack(result, cfg.output_path)

    return result


def _write_haystack(result: HaystackResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.text, encoding="utf-8")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2), encoding="utf-8"
    )


def clues_from_mapping(
    depths: Mapping[str, float],
    texts: Optional[Mapping[str, str]] = None,
    timestamps: Optional[Mapping[str, str]] = None,
) -> List[AnomalyClue]:
    """
    Convenience builder: depths={'A': 10, 'B': 50, 'C': 90}.

    Missing text/timestamps fall back to DEFAULT_CLUES.
    """
    defaults = {c.clue_id: c for c in DEFAULT_CLUES}
    out: List[AnomalyClue] = []
    for clue_id, depth in depths.items():
        base = defaults.get(clue_id)
        out.append(
            AnomalyClue(
                clue_id=clue_id,
                depth_percentile=float(depth),
                text=(texts or {}).get(clue_id) or (base.text if base else f"[{clue_id}]"),
                timestamp_iso=(timestamps or {}).get(clue_id)
                or (base.timestamp_iso if base else "1970-01-01T00:00:00Z"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a long-context telemetry haystack.")
    p.add_argument("--target-tokens", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("artifacts/haystack.txt"))
    p.add_argument("--depth-a", type=float, default=10.0)
    p.add_argument("--depth-b", type=float, default=50.0)
    p.add_argument("--depth-c", type=float, default=90.0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    clues = clues_from_mapping(
        {"A": args.depth_a, "B": args.depth_b, "C": args.depth_c}
    )
    cfg = HaystackConfig(
        target_tokens=args.target_tokens,
        seed=args.seed,
        clues=clues,
        output_path=args.out,
    )
    result = generate_haystack(cfg)
    print(
        json.dumps(
            {
                "approx_token_count": result.approx_token_count,
                "char_count": result.char_count,
                "line_count": result.line_count,
                "content_sha256": result.content_sha256,
                "placements": [asdict(p) for p in result.placements],
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

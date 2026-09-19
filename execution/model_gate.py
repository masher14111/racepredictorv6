"""The Stage-4 model-validation gate, read fail-closed.

Stage 5 must **obey** the Stage-4 verdict rather than re-derive a friendlier one.
This module is the single place that answers "does the model carry a GO?", and it
answers ``False`` unless the audit artifact affirmatively proves otherwise.

Three conditions must all hold for a GO, and each mirrors a lesson the Stage-4
audit paid for (see ``memory/live-repair-04-calibration-audit.md``):

1. **Beats the best de-vig, with an interval.** The headline line's log-loss edge
   over the market must be positive *and* its 95% race-bootstrap CI lower bound
   must be above zero. The comparison is against the **best** de-vig baseline
   (shin/power/proportional), because price-conditioned recalibration is itself
   margin removal — benchmarking it against proportional manufactures edge.
2. **Positive closing-line value, with an interval.** Mean CLV and its CI lower
   bound must both be above zero. A log-loss win with negative CLV is paper-only.
3. **The drift gate passes.** Any feature whose PSI exceeds ``retrain.psi_threshold``
   fails the gate. The gate is the gate.

The headline line is the **independent price-free** line. The market-adjusted line
is reported for completeness but can never grant a GO on its own: Stage 4 measured
it at 0.925 correlation with 1/price, i.e. mostly a price echo.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_VERDICT_PATH = os.path.join("data", "audit", "stage4", "final_evaluation.json")

# The price-free line. Never promote a market-adjusted line to headline.
HEADLINE_LINE = "audit_independent"


@dataclass(frozen=True)
class LineVerdict:
    """One model line's head-to-head result against the de-vigged market."""

    name: str
    n_races: int = 0
    n_runners: int = 0
    model_log_loss: Optional[float] = None
    market_log_loss: Optional[float] = None
    logloss_delta: Optional[float] = None
    logloss_delta_ci95: Optional[tuple[float, float]] = None
    beats_market_logloss: bool = False
    beats_best_devig: bool = False
    model_ece: Optional[float] = None
    market_ece: Optional[float] = None

    @property
    def significant(self) -> bool:
        """Positive edge whose 95% interval excludes zero."""
        if self.logloss_delta is None or self.logloss_delta_ci95 is None:
            return False
        return self.logloss_delta > 0.0 and self.logloss_delta_ci95[0] > 0.0


@dataclass(frozen=True)
class ModelVerdict:
    """The machine-readable Stage-4 verdict Stage 5 consumes."""

    go: bool = False
    source: str = ""
    available: bool = False
    window: Optional[tuple[str, str]] = None
    n_races: int = 0
    n_runners: int = 0
    headline_line: str = HEADLINE_LINE
    lines: dict[str, LineVerdict] = field(default_factory=dict)
    best_devig_method: Optional[str] = None
    best_devig_log_loss: Optional[float] = None
    mean_clv_log: Optional[float] = None
    clv_ci95: Optional[tuple[float, float]] = None
    clv_positive: bool = False
    beat_close_rate: Optional[float] = None
    drift_ok: bool = False
    drift_threshold: float = 0.2
    drift_failures: tuple[tuple[str, float], ...] = ()
    reasons: tuple[str, ...] = ()
    checked_at: str = ""

    @property
    def verdict_label(self) -> str:
        return "GO" if self.go else "NO-GO"

    @property
    def headline(self) -> Optional[LineVerdict]:
        return self.lines.get(self.headline_line)

    def to_dict(self) -> dict:
        return {
            "go": self.go,
            "verdict": self.verdict_label,
            "source": self.source,
            "available": self.available,
            "window": list(self.window) if self.window else None,
            "n_races": self.n_races,
            "n_runners": self.n_runners,
            "headline_line": self.headline_line,
            "lines": {
                name: {
                    "n_races": lv.n_races,
                    "n_runners": lv.n_runners,
                    "model_log_loss": lv.model_log_loss,
                    "market_log_loss": lv.market_log_loss,
                    "logloss_delta": lv.logloss_delta,
                    "logloss_delta_ci95": list(lv.logloss_delta_ci95)
                    if lv.logloss_delta_ci95
                    else None,
                    "beats_market_logloss": lv.beats_market_logloss,
                    "beats_best_devig": lv.beats_best_devig,
                    "significant": lv.significant,
                    "model_ece": lv.model_ece,
                    "market_ece": lv.market_ece,
                }
                for name, lv in self.lines.items()
            },
            "best_devig_method": self.best_devig_method,
            "best_devig_log_loss": self.best_devig_log_loss,
            "mean_clv_log": self.mean_clv_log,
            "clv_ci95": list(self.clv_ci95) if self.clv_ci95 else None,
            "clv_positive": self.clv_positive,
            "beat_close_rate": self.beat_close_rate,
            "drift_ok": self.drift_ok,
            "drift_threshold": self.drift_threshold,
            "drift_failures": [list(f) for f in self.drift_failures],
            "reasons": list(self.reasons),
            "checked_at": self.checked_at,
        }


def _ci(raw: Any) -> Optional[tuple[float, float]]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
    return None


def _f(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _no_go(source: str, reason: str, **kw: Any) -> ModelVerdict:
    logger.warning("model gate: NO-GO (%s)", reason)
    return ModelVerdict(
        go=False, source=source, reasons=(reason,), checked_at=_now_iso(), **kw
    )


def load_model_verdict(
    path: Optional[str] = None, *, raw_config: Optional[dict] = None
) -> ModelVerdict:
    """Read the Stage-4 audit artifact and derive the GO/NO-GO verdict.

    Fail-closed at every step: a missing file, unparseable JSON, a missing
    headline line, or an absent confidence interval all yield NO-GO.
    """
    source = path or DEFAULT_VERDICT_PATH
    try:
        cfg = raw_config if raw_config is not None else get_config()
    except Exception:  # pragma: no cover - config always loads in practice
        cfg = {}
    drift_threshold = _f((cfg.get("retrain") or {}).get("psi_threshold")) or 0.2

    if not os.path.exists(source):
        return _no_go(source, f"audit_artifact_missing:{source}")
    try:
        with open(source, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return _no_go(source, f"audit_artifact_unreadable:{exc}")
    if not isinstance(payload, dict):
        return _no_go(source, "audit_artifact_malformed")

    reasons: list[str] = []

    # ── de-vig baseline: compare against the BEST, never proportional ─────────
    baselines = payload.get("market_devig_method_baselines") or {}
    best_method: Optional[str] = None
    best_ll: Optional[float] = None
    for method, node in baselines.items():
        if not isinstance(node, dict):
            continue
        ll = _f(node.get("race_log_loss"))
        if ll is None:
            continue
        if best_ll is None or ll < best_ll:
            best_ll, best_method = ll, method

    # ── per-line head-to-head ────────────────────────────────────────────────
    h2h = payload.get("head_to_head") or {}
    lines: dict[str, LineVerdict] = {}
    for name, node in h2h.items():
        if not isinstance(node, dict):
            continue
        model_ll = _f(node.get("model_log_loss"))
        delta = _f(node.get("logloss_delta_market_minus_model"))
        ci = _ci(node.get("logloss_delta_ci95"))
        beats_best = bool(
            model_ll is not None and best_ll is not None and model_ll < best_ll
        )
        lines[name] = LineVerdict(
            name=name,
            n_races=int(node.get("n_races") or 0),
            n_runners=int(node.get("n_runners") or 0),
            model_log_loss=model_ll,
            market_log_loss=_f(node.get("market_log_loss")),
            logloss_delta=delta,
            logloss_delta_ci95=ci,
            beats_market_logloss=bool(node.get("model_beats_market_logloss")),
            beats_best_devig=beats_best,
            model_ece=_f(node.get("model_ece")),
            market_ece=_f(node.get("market_ece")),
        )

    headline = lines.get(HEADLINE_LINE)
    if headline is None:
        reasons.append(f"headline_line_missing:{HEADLINE_LINE}")
    else:
        if headline.logloss_delta_ci95 is None:
            reasons.append("logloss_delta_ci95_missing")
        elif not headline.significant:
            lo, hi = headline.logloss_delta_ci95
            reasons.append(
                f"logloss_edge_not_significant:delta={headline.logloss_delta:+.5f} "
                f"ci95=[{lo:+.5f},{hi:+.5f}]"
            )
        if not headline.beats_best_devig:
            reasons.append(
                "loses_to_best_devig:"
                f"{best_method or 'unknown'}={best_ll if best_ll is not None else float('nan'):.5f}"
            )

    # ── closing-line value ───────────────────────────────────────────────────
    clv_node = (payload.get("clv") or {}).get("all_common_rows") or {}
    mean_clv = _f(clv_node.get("mean_clv_log"))
    clv_ci = _ci(clv_node.get("mean_clv_ci95"))
    clv_positive = bool(
        mean_clv is not None and mean_clv > 0.0 and clv_ci is not None and clv_ci[0] > 0.0
    )
    if mean_clv is None:
        reasons.append("clv_missing")
    elif not clv_positive:
        lo = clv_ci[0] if clv_ci else float("nan")
        reasons.append(f"clv_not_positive:mean={mean_clv:+.5f} ci95_lower={lo:+.5f}")

    # ── drift ────────────────────────────────────────────────────────────────
    psi_rows = (payload.get("drift") or {}).get("psi_by_feature") or []
    failures: list[tuple[str, float]] = []
    for row in psi_rows:
        if not isinstance(row, dict):
            continue
        psi = _f(row.get("psi"))
        if psi is not None and psi > drift_threshold:
            failures.append((str(row.get("feature")), psi))
    drift_ok = not failures
    if not drift_ok:
        worst = ", ".join(f"{f}={p:.3f}" for f, p in failures[:3])
        reasons.append(f"drift_gate_failed:{worst}")

    window_node = payload.get("window") or {}
    window = None
    if window_node.get("start") and window_node.get("end"):
        window = (str(window_node["start"]), str(window_node["end"]))

    go = not reasons
    verdict = ModelVerdict(
        go=go,
        source=source,
        available=True,
        window=window,
        n_races=int(payload.get("n_common_races") or 0),
        n_runners=int(payload.get("n_common_rows") or 0),
        lines=lines,
        best_devig_method=best_method,
        best_devig_log_loss=best_ll,
        mean_clv_log=mean_clv,
        clv_ci95=clv_ci,
        clv_positive=clv_positive,
        beat_close_rate=_f(clv_node.get("beat_close_rate")),
        drift_ok=drift_ok,
        drift_threshold=drift_threshold,
        drift_failures=tuple(failures),
        reasons=tuple(reasons),
        checked_at=_now_iso(),
    )
    logger.info(
        "model gate: %s (%d reasons) from %s",
        verdict.verdict_label,
        len(reasons),
        source,
    )
    return verdict


def model_go(path: Optional[str] = None) -> bool:
    """Shorthand for ``load_model_verdict(path).go``."""
    return load_model_verdict(path).go

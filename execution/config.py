"""The ``execution:`` config contract for Stage 5.

Every number here is a **risk ceiling, not a profitability claim**. Two
mechanisms keep it that way:

* **Provisional clamps.** :data:`PROVISIONAL_CEILINGS` caps the staking and
  loss-limit knobs at their documented provisional values. ``from_config`` clamps
  anything looser and records what it clamped in ``clamps_applied`` so the
  tightening is reported, never silent. The clamps are only lifted by
  :meth:`ExecutionConfig.released`, which demands a model GO *and* a passing
  forward gate *and* ``paper_only: false``.
* **Gate floors.** :data:`FORWARD_GATE_FLOORS` stops the forward-release criteria
  being loosened below the values Stage 5 committed to (>= 8 weeks, a meaningful
  qualified sample, a CLV interval strictly above zero). A config that tries to
  loosen them is raised back to the floor and the change is recorded.

Both directions fail closed: an absent ``execution:`` block yields the strictest
possible configuration with ``paper_only=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Staking / loss knobs may never be looser than these while the deployment is
# provisional. Keys are ``"<section>.<field>"``; the value is the loosest
# permitted magnitude (all of these are "smaller is safer").
PROVISIONAL_CEILINGS: dict[str, float] = {
    "staking.kelly_fraction": 0.10,
    "staking.max_stake_pct_bankroll": 0.005,
    "staking.max_race_exposure_pct": 0.005,
    "staking.max_daily_exposure_pct": 0.03,
    "safeguards.bankroll_stop_loss_pct": 0.20,
    "safeguards.daily_loss_limit_pct": 0.02,
}

# Forward-release criteria may never be *weaker* than these. ``"min"`` entries
# are lower bounds the config must meet or exceed; ``"max"`` entries are upper
# bounds it may not exceed.
FORWARD_GATE_FLOORS: dict[str, tuple[str, float]] = {
    "min_weeks": ("min", 8.0),
    "min_qualified_bets": ("min", 200.0),
    "min_qualified_races": ("min", 150.0),
    "clv_ci_lower_above": ("min", 0.0),
    "max_calibration_ece": ("max", 0.03),
    "max_drawdown_pct": ("max", 0.10),
}

_DEFAULT_EXCHANGE_SOURCES = ("betfair", "betdaq", "smarkets", "matchbook")


def _as_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _as_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def _section(raw: dict, name: str) -> dict:
    node = raw.get(name)
    return node if isinstance(node, dict) else {}


@dataclass(frozen=True)
class SnapshotConfig:
    """Point-in-time odds snapshot store (requirement 1)."""

    enabled: bool = True
    db_path: str = "data/races.db"
    max_age_seconds: float = 900.0
    retention_days: int = 400

    @classmethod
    def from_section(cls, sec: dict, *, default_db_path: str) -> "SnapshotConfig":
        return cls(
            enabled=_as_bool(sec.get("enabled"), True),
            db_path=str(sec.get("db_path") or default_db_path),
            max_age_seconds=_as_float(sec.get("max_age_seconds"), 900.0),
            retention_days=_as_int(sec.get("retention_days"), 400),
        )


@dataclass(frozen=True)
class FrictionConfig:
    """Execution frictions the walk-forward simulator applies (requirement 2)."""

    decision_latency_seconds: float = 30.0
    placement_latency_seconds: float = 15.0
    max_price_age_seconds: float = 300.0
    allow_stale_fill: bool = False
    rejection_rate: float = 0.05
    suspension_rate: float = 0.02
    adverse_move_only: bool = False
    max_stake_per_bet: float = 50.0
    exchange_commission: float = 0.02
    exchange_sources: tuple[str, ...] = _DEFAULT_EXCHANGE_SOURCES
    best_odds_guaranteed: bool = False
    rule_4_enabled: bool = True
    dead_heat_enabled: bool = True
    void_non_runners: bool = True

    @property
    def total_latency_seconds(self) -> float:
        """Seconds between the model emitting a candidate and the book seeing it."""
        return max(0.0, self.decision_latency_seconds) + max(
            0.0, self.placement_latency_seconds
        )

    def is_exchange(self, source: Optional[str]) -> bool:
        if not source:
            return False
        return str(source).strip().lower() in self.exchange_sources

    def commission_for(self, source: Optional[str]) -> float:
        """Commission rate on net winnings for ``source`` (0.0 for bookmakers)."""
        return self.exchange_commission if self.is_exchange(source) else 0.0

    @classmethod
    def from_section(cls, sec: dict) -> "FrictionConfig":
        raw_sources = sec.get("exchange_sources")
        if isinstance(raw_sources, (list, tuple)):
            sources = tuple(
                str(s).strip().lower() for s in raw_sources if str(s).strip()
            )
        else:
            sources = _DEFAULT_EXCHANGE_SOURCES
        return cls(
            decision_latency_seconds=_as_float(sec.get("decision_latency_seconds"), 30.0),
            placement_latency_seconds=_as_float(sec.get("placement_latency_seconds"), 15.0),
            max_price_age_seconds=_as_float(sec.get("max_price_age_seconds"), 300.0),
            allow_stale_fill=_as_bool(sec.get("allow_stale_fill"), False),
            rejection_rate=_as_float(sec.get("rejection_rate"), 0.05),
            suspension_rate=_as_float(sec.get("suspension_rate"), 0.02),
            adverse_move_only=_as_bool(sec.get("adverse_move_only"), False),
            max_stake_per_bet=_as_float(sec.get("max_stake_per_bet"), 50.0),
            exchange_commission=_as_float(sec.get("exchange_commission"), 0.02),
            exchange_sources=sources,
            # Never default BOG on: it must be evidenced per bet, not assumed.
            best_odds_guaranteed=_as_bool(sec.get("best_odds_guaranteed"), False),
            rule_4_enabled=_as_bool(sec.get("rule_4_enabled"), True),
            dead_heat_enabled=_as_bool(sec.get("dead_heat_enabled"), True),
            void_non_runners=_as_bool(sec.get("void_non_runners"), True),
        )


@dataclass(frozen=True)
class StakingConfig:
    """Provisional staking ceilings (requirement 6).

    The exposure caps bind *before* Kelly, so the Kelly fraction can only ever
    shrink a stake. ``kelly_fraction`` at 0.10 is a provisional setting for an
    unvalidated model, not a Kelly-optimal choice.
    """

    kelly_fraction: float = 0.10
    max_stake_pct_bankroll: float = 0.005
    max_race_exposure_pct: float = 0.005
    max_daily_exposure_pct: float = 0.03
    min_stake: float = 0.50
    stake_rounding: float = 0.10
    allow_accumulators: bool = False
    allow_correlated_bets: bool = False

    @classmethod
    def from_section(cls, sec: dict) -> "StakingConfig":
        return cls(
            kelly_fraction=_as_float(sec.get("kelly_fraction"), 0.10),
            max_stake_pct_bankroll=_as_float(sec.get("max_stake_pct_bankroll"), 0.005),
            max_race_exposure_pct=_as_float(sec.get("max_race_exposure_pct"), 0.005),
            max_daily_exposure_pct=_as_float(sec.get("max_daily_exposure_pct"), 0.03),
            min_stake=_as_float(sec.get("min_stake"), 0.50),
            stake_rounding=_as_float(sec.get("stake_rounding"), 0.10),
            # Multiples and same-race correlated bets are never enabled by config.
            allow_accumulators=_as_bool(sec.get("allow_accumulators"), False),
            allow_correlated_bets=_as_bool(sec.get("allow_correlated_bets"), False),
        )


@dataclass(frozen=True)
class SafeguardConfig:
    """Bankroll and issuance safeguards (requirement 9)."""

    bankroll_stop_loss_pct: float = 0.20
    daily_loss_limit_pct: float = 0.02
    block_started_races: bool = True
    started_race_buffer_seconds: int = 60
    block_duplicates: bool = True
    block_stale_sources: bool = True
    max_source_age_seconds: float = 900.0
    max_open_tickets: int = 25

    @classmethod
    def from_section(cls, sec: dict) -> "SafeguardConfig":
        return cls(
            bankroll_stop_loss_pct=_as_float(sec.get("bankroll_stop_loss_pct"), 0.20),
            daily_loss_limit_pct=_as_float(sec.get("daily_loss_limit_pct"), 0.02),
            block_started_races=_as_bool(sec.get("block_started_races"), True),
            started_race_buffer_seconds=_as_int(sec.get("started_race_buffer_seconds"), 60),
            block_duplicates=_as_bool(sec.get("block_duplicates"), True),
            block_stale_sources=_as_bool(sec.get("block_stale_sources"), True),
            max_source_age_seconds=_as_float(sec.get("max_source_age_seconds"), 900.0),
            max_open_tickets=_as_int(sec.get("max_open_tickets"), 25),
        )


@dataclass(frozen=True)
class GateConfig:
    """Candidate gate — PASS unless every condition is met (requirement 5)."""

    require_complete_card: bool = True
    require_runner_history: bool = True
    require_calibrated_probability: bool = True
    require_reference_market: bool = True
    require_executable_price: bool = True
    require_source_health: bool = True
    require_model_validation: bool = True
    min_edge: float = 0.02
    min_expected_value: float = 0.05
    min_field_size: int = 5
    max_overround: float = 1.25

    @classmethod
    def from_section(cls, sec: dict) -> "GateConfig":
        return cls(
            require_complete_card=_as_bool(sec.get("require_complete_card"), True),
            require_runner_history=_as_bool(sec.get("require_runner_history"), True),
            require_calibrated_probability=_as_bool(
                sec.get("require_calibrated_probability"), True
            ),
            require_reference_market=_as_bool(sec.get("require_reference_market"), True),
            require_executable_price=_as_bool(sec.get("require_executable_price"), True),
            require_source_health=_as_bool(sec.get("require_source_health"), True),
            require_model_validation=_as_bool(sec.get("require_model_validation"), True),
            min_edge=_as_float(sec.get("min_edge"), 0.02),
            min_expected_value=_as_float(sec.get("min_expected_value"), 0.05),
            min_field_size=_as_int(sec.get("min_field_size"), 5),
            max_overround=_as_float(sec.get("max_overround"), 1.25),
        )


@dataclass(frozen=True)
class SelectionConfig:
    """Anti-threshold-mining protocol (requirement 4)."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    max_strategies: int = 64
    correction: str = "sidak"
    alpha: float = 0.05
    seed: int = 20260727
    bootstrap_resamples: int = 1000
    lock_file: str = "data/execution/selection_lock.json"

    @property
    def test_fraction(self) -> float:
        """The untouched window. Scored exactly once, after selection is locked."""
        return max(0.0, 1.0 - self.train_fraction - self.validation_fraction)

    @classmethod
    def from_section(cls, sec: dict) -> "SelectionConfig":
        correction = str(sec.get("correction") or "sidak").strip().lower()
        if correction not in ("bonferroni", "sidak", "none"):
            logger.warning(
                "execution.selection.correction=%r unknown; using 'sidak'", correction
            )
            correction = "sidak"
        return cls(
            train_fraction=_as_float(sec.get("train_fraction"), 0.6),
            validation_fraction=_as_float(sec.get("validation_fraction"), 0.2),
            max_strategies=_as_int(sec.get("max_strategies"), 64),
            correction=correction,
            alpha=_as_float(sec.get("alpha"), 0.05),
            seed=_as_int(sec.get("seed"), 20260727),
            bootstrap_resamples=_as_int(sec.get("bootstrap_resamples"), 1000),
            lock_file=str(sec.get("lock_file") or "data/execution/selection_lock.json"),
        )


@dataclass(frozen=True)
class ForwardGateConfig:
    """Forward-release criteria (requirement 7).

    Satisfiable by **forward** (shadow/paper) evidence only. A backtest, however
    long or however profitable, can never clear this gate.
    """

    min_weeks: float = 8.0
    min_qualified_bets: int = 200
    min_qualified_races: int = 150
    require_positive_mean_clv: bool = True
    clv_ci_lower_above: float = 0.0
    ae_min: float = 0.90
    ae_max: float = 1.10
    max_calibration_ece: float = 0.03
    max_drawdown_pct: float = 0.10
    require_model_go: bool = True

    @classmethod
    def from_section(cls, sec: dict) -> "ForwardGateConfig":
        return cls(
            min_weeks=_as_float(sec.get("min_weeks"), 8.0),
            min_qualified_bets=_as_int(sec.get("min_qualified_bets"), 200),
            min_qualified_races=_as_int(sec.get("min_qualified_races"), 150),
            require_positive_mean_clv=_as_bool(sec.get("require_positive_mean_clv"), True),
            clv_ci_lower_above=_as_float(sec.get("clv_ci_lower_above"), 0.0),
            ae_min=_as_float(sec.get("ae_min"), 0.90),
            ae_max=_as_float(sec.get("ae_max"), 1.10),
            max_calibration_ece=_as_float(sec.get("max_calibration_ece"), 0.03),
            max_drawdown_pct=_as_float(sec.get("max_drawdown_pct"), 0.10),
            require_model_go=_as_bool(sec.get("require_model_go"), True),
        )


@dataclass(frozen=True)
class ReportConfig:
    """Where dated reports and machine-readable artifacts land (requirement 10)."""

    dir: str = "reports"
    artifact_dir: str = "data/execution"
    currency: str = "€"

    @classmethod
    def from_section(cls, sec: dict) -> "ReportConfig":
        return cls(
            dir=str(sec.get("dir") or "reports"),
            artifact_dir=str(sec.get("artifact_dir") or "data/execution"),
            currency=str(sec.get("currency") or "€"),
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """The whole ``execution:`` block, loaded fail-closed."""

    paper_only: bool = True
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    frictions: FrictionConfig = field(default_factory=FrictionConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    safeguards: SafeguardConfig = field(default_factory=SafeguardConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    forward_gate: ForwardGateConfig = field(default_factory=ForwardGateConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    # Human-readable record of every value this loader tightened, so a report can
    # show that a loosened config did not take effect.
    clamps_applied: tuple[str, ...] = ()
    provisional: bool = True

    # ── loading ──────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, raw: Optional[dict] = None) -> "ExecutionConfig":
        """Build from the merged config.yaml ``execution:`` block.

        A missing block is not an error: it yields the strictest configuration
        (``paper_only=True`` with every default ceiling in force).
        """
        raw = raw if raw is not None else get_config()
        ex = _section(raw, "execution")
        default_db = str(raw.get("database_path") or "data/races.db")

        cfg = cls(
            # Fail closed: anything other than an explicit false leaves paper-only on.
            paper_only=_as_bool(ex.get("paper_only"), True),
            snapshots=SnapshotConfig.from_section(
                _section(ex, "snapshots"), default_db_path=default_db
            ),
            frictions=FrictionConfig.from_section(_section(ex, "frictions")),
            staking=StakingConfig.from_section(_section(ex, "staking")),
            safeguards=SafeguardConfig.from_section(_section(ex, "safeguards")),
            gates=GateConfig.from_section(_section(ex, "gates")),
            selection=SelectionConfig.from_section(_section(ex, "selection")),
            forward_gate=ForwardGateConfig.from_section(_section(ex, "forward_gate")),
            reports=ReportConfig.from_section(_section(ex, "reports")),
        )
        return cfg._apply_provisional_limits()

    def _apply_provisional_limits(self) -> "ExecutionConfig":
        """Clamp staking/loss knobs down and forward-gate criteria back up."""
        clamps: list[str] = []

        staking_updates: dict[str, float] = {}
        safeguard_updates: dict[str, float] = {}
        for key, ceiling in PROVISIONAL_CEILINGS.items():
            section_name, attr = key.split(".", 1)
            section = getattr(self, section_name)
            value = float(getattr(section, attr))
            if value > ceiling:
                clamps.append(f"{key}: {value:g} -> {ceiling:g} (provisional ceiling)")
                if section_name == "staking":
                    staking_updates[attr] = ceiling
                else:
                    safeguard_updates[attr] = ceiling

        gate_updates: dict[str, float] = {}
        for attr, (kind, bound) in FORWARD_GATE_FLOORS.items():
            value = float(getattr(self.forward_gate, attr))
            if kind == "min" and value < bound:
                clamps.append(
                    f"forward_gate.{attr}: {value:g} -> {bound:g} (criterion floor)"
                )
                gate_updates[attr] = bound
            elif kind == "max" and value > bound:
                clamps.append(
                    f"forward_gate.{attr}: {value:g} -> {bound:g} (criterion ceiling)"
                )
                gate_updates[attr] = bound
        if not self.forward_gate.require_positive_mean_clv:
            clamps.append("forward_gate.require_positive_mean_clv: false -> true")
        if not self.forward_gate.require_model_go:
            clamps.append("forward_gate.require_model_go: false -> true")
        # The candidate gate's model-validation condition is the same precondition
        # one layer earlier. ``execution.gates`` promises there is "deliberately no
        # override" that issues a candidate under a NO-GO verdict — a YAML flag must
        # not be that override (step 17: one line issued a CANDIDATE under NO-GO).
        if not self.gates.require_model_validation:
            clamps.append("gates.require_model_validation: false -> true")

        # Multiples / correlated bets are structurally disallowed while provisional.
        if self.staking.allow_accumulators:
            clamps.append("staking.allow_accumulators: true -> false")
        if self.staking.allow_correlated_bets:
            clamps.append("staking.allow_correlated_bets: true -> false")

        if not clamps:
            return replace(self, provisional=True, clamps_applied=())

        for line in clamps:
            logger.warning("execution config tightened — %s", line)

        staking = replace(
            self.staking,
            allow_accumulators=False,
            allow_correlated_bets=False,
            **staking_updates,
        )
        safeguards = (
            replace(self.safeguards, **safeguard_updates)
            if safeguard_updates
            else self.safeguards
        )
        forward_gate = replace(
            self.forward_gate,
            require_positive_mean_clv=True,
            require_model_go=True,
            **{k: (int(v) if isinstance(getattr(self.forward_gate, k), int) else v)
               for k, v in gate_updates.items()},
        )
        return replace(
            self,
            staking=staking,
            safeguards=safeguards,
            gates=replace(self.gates, require_model_validation=True),
            forward_gate=forward_gate,
            provisional=True,
            clamps_applied=tuple(clamps),
        )

    # ── release ──────────────────────────────────────────────────────────────
    def released(self, *, model_go: bool, forward_gate_passed: bool) -> "ExecutionConfig":
        """Return a non-provisional copy, but only when the evidence supports it.

        Both a Stage-4-style model GO and a passing forward gate are required, on
        top of ``paper_only: false``. Anything less returns ``self`` unchanged —
        there is no code path that lifts a provisional ceiling without evidence.
        """
        if self.paper_only or not model_go or not forward_gate_passed:
            return self
        return replace(self, provisional=False)

    # ── derived state ────────────────────────────────────────────────────────
    def deployment_mode(self, *, model_go: bool, forward_gate_passed: bool) -> str:
        """``"PAPER-ONLY"`` or ``"CANDIDATE-ELIGIBLE"``."""
        return (
            "CANDIDATE-ELIGIBLE"
            if real_money_enabled(
                self, model_go=model_go, forward_gate_passed=forward_gate_passed
            )
            else "PAPER-ONLY"
        )

    def artifact_path(self, *parts: str) -> str:
        return os.path.join(self.reports.artifact_dir, *parts)

    def report_path(self, *parts: str) -> str:
        return os.path.join(self.reports.dir, *parts)

    def to_dict(self) -> dict:
        """Plain-dict view for reports and JSON artifacts."""

        def _d(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _d(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        return _d(self)


def real_money_enabled(
    cfg: ExecutionConfig, *, model_go: bool, forward_gate_passed: bool
) -> bool:
    """The single place that may ever answer "show real-money recommendations".

    All three must hold: the permanent paper-only override is off, the model
    carries a GO verdict, and the forward-release gate has passed on forward
    evidence. Stage 4's MODEL NO-GO alone keeps this ``False``.
    """
    return bool(
        cfg is not None
        and not cfg.paper_only
        and model_go
        and forward_gate_passed
    )


def load_execution_config(raw: Optional[dict] = None) -> ExecutionConfig:
    """Convenience alias mirroring the other config loaders in this repo."""
    return ExecutionConfig.from_config(raw)


def config_fingerprint(cfg: ExecutionConfig) -> str:
    """A short, stable hash of every risk/gate value this config actually holds.

    Stage 20 (B7) provenance: a ticket records this alongside the model bundle's
    own content hash, so "which configuration produced this decision" is
    answerable without diffing YAML by hand. Hashes ``to_dict()`` — the same
    plain-dict view already used for reports — with sorted keys, so field order
    and float/str repr differences never change the fingerprint spuriously.
    """
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe_clamps(cfg: ExecutionConfig) -> Sequence[str]:
    """Lines a report can print verbatim to show what the loader tightened."""
    return cfg.clamps_applied

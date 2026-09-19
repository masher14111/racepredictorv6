"""Bet tracking: stake sizing, recording, settlement, and P&L aggregation."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from utils.logger import get_logger
from utils.notifications import get_notifier
from utils.storage import DEFAULT_DB_PATH
from utils.storage.migrations import apply_migrations
from utils.storage.pool import ConnectionPool
from utils.timezone import now, to_utc

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "bets.csv"


class Strategy(str, Enum):
    FLAT = "flat"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"


class StopLossError(RuntimeError):
    """Raised when a bet is attempted while the stop-loss floor is active."""


class DuplicateBetError(RuntimeError):
    """Raised when a paper bet duplicates an existing bet on the same selection."""


class RaceStartedError(RuntimeError):
    """Raised when a paper bet is attempted on a race that has already started."""


class BetTracker:
    """
    Record bets, recommend stakes via flat/Kelly strategies, and aggregate P&L.

    Parameters
    ----------
    db_path:
        SQLite database path. Defaults to data/races.db.
    initial_bankroll:
        Starting balance used only when no prior bankroll_log exists.
    flat_stake:
        Fixed stake per bet for the FLAT strategy.
    kelly_fraction:
        Fraction of full Kelly applied by FRACTIONAL_KELLY (e.g. 0.25 = quarter Kelly).
    stop_loss_pct:
        Stop all new bets when bankroll drops below initial_bankroll * (1 − stop_loss_pct).
    ew_fraction:
        Each-way place fraction (1/5 = 0.20 is standard UK/Ireland horse racing).
    ew_places:
        Number of places paid on each-way bets (informational; affects settlement).
    """

    def __init__(
        self,
        db_path: str | None = None,
        initial_bankroll: float = 1000.0,
        flat_stake: float = 10.0,
        kelly_fraction: float = 0.25,
        stop_loss_pct: float = 0.20,
        ew_fraction: float = 0.20,
        ew_places: int = 3,
    ) -> None:
        self._pool = ConnectionPool(db_path or DEFAULT_DB_PATH)
        apply_migrations(self._pool)
        self._initial_bankroll = initial_bankroll
        self._flat_stake = flat_stake
        self._kelly_fraction = kelly_fraction
        self._stop_loss_pct = stop_loss_pct
        self._ew_fraction = ew_fraction
        self._ew_places = ew_places
        self._ensure_bankroll()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        return self._pool.connection()

    def _ensure_bankroll(self) -> None:
        row = self._conn().execute(
            "SELECT id FROM bankroll_log LIMIT 1"
        ).fetchone()
        if row is None:
            with self._pool.write_lock():
                conn = self._conn()
                conn.execute(
                    "INSERT INTO bankroll_log(logged_at, event_type, amount, balance) "
                    "VALUES (?,?,?,?)",
                    (now().isoformat(), "init", self._initial_bankroll, self._initial_bankroll),
                )
                conn.commit()
            logger.info("bet_tracker: bankroll initialised at %.2f", self._initial_bankroll)

    # ------------------------------------------------------------------
    # Bankroll state
    # ------------------------------------------------------------------

    @property
    def bankroll(self) -> float:
        row = self._conn().execute(
            "SELECT balance FROM bankroll_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return float(row["balance"]) if row else self._initial_bankroll

    @property
    def initial_bankroll(self) -> float:
        row = self._conn().execute(
            "SELECT balance FROM bankroll_log WHERE event_type='init' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return float(row["balance"]) if row else self._initial_bankroll

    @property
    def ew_places(self) -> int:
        return self._ew_places

    def is_stopped(self) -> bool:
        """True when the bankroll has fallen strictly below the stop-loss floor."""
        floor = self.initial_bankroll * (1.0 - self._stop_loss_pct)
        return self.bankroll < floor

    def set_bankroll(self, amount: float) -> None:
        """Set the virtual bankroll to ``amount``.

        Before any bets are placed this resets the baseline (so ROI and drawdown
        measure from the new figure); once bets exist it records an adjustment
        event, keeping the equity curve continuous.
        """
        if amount <= 0:
            raise ValueError("bankroll must be positive")
        with self._pool.write_lock():
            conn = self._conn()
            n_bets = conn.execute("SELECT COUNT(*) AS c FROM bets").fetchone()["c"]
            if n_bets == 0:
                conn.execute(
                    "UPDATE bankroll_log SET amount=?, balance=? WHERE event_type='init'",
                    (amount, amount),
                )
            else:
                delta = amount - self.bankroll
                conn.execute(
                    "INSERT INTO bankroll_log(logged_at, event_type, amount, balance) "
                    "VALUES (?,?,?,?)",
                    (now().isoformat(), "adjust", delta, amount),
                )
            conn.commit()
        logger.info("bet_tracker: bankroll set to %.2f", amount)

    # ------------------------------------------------------------------
    # Stake sizing
    # ------------------------------------------------------------------

    def kelly_stake(self, odds_decimal: float, win_prob: float) -> float:
        """Full Kelly stake in currency units (not fraction). Returns 0 if no edge."""
        b = odds_decimal - 1.0
        if b <= 0.0 or win_prob <= 0.0 or win_prob >= 1.0:
            return 0.0
        q = 1.0 - win_prob
        f = (b * win_prob - q) / b
        return max(0.0, f) * self.bankroll

    def recommend_stake(
        self,
        odds_decimal: float,
        win_prob: float,
        strategy: Strategy | str = Strategy.FLAT,
    ) -> float:
        """
        Recommended stake for a bet. Returns 0.0 if stop-loss is active or there is no edge.

        strategy:
            'flat'             — always return flat_stake
            'kelly'            — full Kelly stake based on current bankroll
            'fractional_kelly' — kelly_fraction × full Kelly
        """
        if self.is_stopped():
            return 0.0
        s = Strategy(strategy)
        if s == Strategy.FLAT:
            return self._flat_stake
        full = self.kelly_stake(odds_decimal, win_prob)
        if s == Strategy.KELLY:
            return round(full, 2)
        return round(full * self._kelly_fraction, 2)

    # ------------------------------------------------------------------
    # Recording bets
    # ------------------------------------------------------------------

    def record_bet(
        self,
        horse_name: str,
        odds_decimal: float,
        stake: float,
        bet_type: str = "win",
        race_id: str | None = None,
        horse_id: str | None = None,
        venue: str | None = None,
        race_time: str | None = None,
        composite_score: float | None = None,
        won_prob: float | None = None,
        value_edge: float | None = None,
        strategy: Strategy | str = Strategy.FLAT,
        notes: str | None = None,
    ) -> int:
        """
        Insert a pending bet and deduct the stake from the bankroll.

        ``won_prob`` and ``value_edge`` capture the model's probability and edge
        at bet time so closing-line value can be measured at settlement.

        Returns the new bet ID.
        Raises StopLossError if the stop-loss floor is active.
        """
        if self.is_stopped():
            raise StopLossError(
                f"stop-loss active — bankroll {self.bankroll:.2f} is at or below "
                f"floor {self.initial_bankroll * (1 - self._stop_loss_pct):.2f}"
            )
        if bet_type not in ("win", "each_way"):
            raise ValueError(f"bet_type must be 'win' or 'each_way', got '{bet_type}'")
        strategy = Strategy(strategy)
        placed_at = now().isoformat()
        with self._pool.write_lock():
            # Read the balance inside the write lock so two concurrent bets cannot
            # both deduct from the same pre-bet bankroll (TOCTOU overcrediting).
            bankroll_before = self.bankroll
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO bets("
                "  placed_at, race_id, horse_id, horse_name, venue, race_time,"
                "  bet_type, stake, odds_decimal, composite_score, won_prob,"
                "  value_edge, strategy, bankroll_before, notes"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    placed_at, race_id, horse_id, horse_name, venue, race_time,
                    bet_type, stake, odds_decimal, composite_score, won_prob,
                    value_edge, strategy.value, bankroll_before, notes,
                ),
            )
            bet_id = cur.lastrowid
            new_balance = bankroll_before - stake
            conn.execute(
                "INSERT INTO bankroll_log(logged_at, event_type, bet_id, amount, balance) "
                "VALUES (?,?,?,?,?)",
                (placed_at, "bet_placed", bet_id, -stake, new_balance),
            )
            conn.commit()
        logger.info(
            "bet_tracker: bet #%d %s @ %.2f stake=%.2f bankroll %.2f→%.2f",
            bet_id, horse_name, odds_decimal, stake, bankroll_before, new_balance,
        )
        floor = self.initial_bankroll * (1.0 - self._stop_loss_pct)
        if new_balance < floor:
            get_notifier().notify_stop_loss(new_balance, floor)
        return bet_id

    # ------------------------------------------------------------------
    # Paper-betting guards
    # ------------------------------------------------------------------

    def has_open_bet(self, race_id, horse_id, bet_type: str = "win") -> bool:
        """True if a bet on this race/horse/market already exists.

        Guards against double-betting. Only fires when both ids are supplied
        (free-form bets with no race context are never treated as duplicates).
        A win and an each-way bet on the same runner are distinct markets.
        """
        if not race_id or not horse_id:
            return False
        row = self._conn().execute(
            "SELECT 1 FROM bets WHERE race_id=? AND horse_id=? AND bet_type=? LIMIT 1",
            (str(race_id), str(horse_id), bet_type),
        ).fetchone()
        return row is not None

    @staticmethod
    def race_has_started(race_time: str | None, _now: datetime | None = None) -> bool:
        """True if ``race_time`` (ISO string) is at or before now. Unknown/blank
        times return False so a missing post-time never blocks a paper bet."""
        if not race_time:
            return False
        try:
            dt = datetime.fromisoformat(race_time)
        except (ValueError, TypeError):
            return False
        ref = _now or now()
        return to_utc(dt) <= to_utc(ref)

    def place_paper_bet(
        self,
        horse_name: str,
        odds_decimal: float,
        stake: float,
        bet_type: str = "win",
        race_id: str | None = None,
        horse_id: str | None = None,
        venue: str | None = None,
        race_time: str | None = None,
        composite_score: float | None = None,
        won_prob: float | None = None,
        value_edge: float | None = None,
        strategy: Strategy | str = Strategy.FLAT,
        notes: str | None = None,
        _now: datetime | None = None,
    ) -> int:
        """Record a paper bet, guarding against started races and double-betting.

        Raises RaceStartedError, DuplicateBetError, or StopLossError.
        Otherwise delegates to :meth:`record_bet`.
        """
        if self.race_has_started(race_time, _now):
            raise RaceStartedError(
                f"race at {race_time} has already started — no bets after the off"
            )
        if self.has_open_bet(race_id, horse_id, bet_type):
            raise DuplicateBetError(
                f"already have a {bet_type} bet on {horse_name} in this race"
            )
        return self.record_bet(
            horse_name, odds_decimal, stake, bet_type=bet_type,
            race_id=race_id, horse_id=horse_id, venue=venue, race_time=race_time,
            composite_score=composite_score, won_prob=won_prob, value_edge=value_edge,
            strategy=strategy, notes=notes,
        )

    # ------------------------------------------------------------------
    # Settling bets
    # ------------------------------------------------------------------

    def _calc_gross_return(
        self, bet_type: str, stake: float, odds_decimal: float, outcome: str
    ) -> float:
        """
        Gross return (stake + profit on winning legs).

        Each-way settlement: win leg at full odds, place leg at
        (odds_decimal − 1) × ew_fraction + 1. Each leg is half the total stake.
        """
        if bet_type == "win":
            return stake * odds_decimal if outcome == "win" else 0.0
        half = stake / 2.0
        place_odds = (odds_decimal - 1.0) * self._ew_fraction + 1.0
        win_return = half * odds_decimal if outcome == "win" else 0.0
        place_return = half * place_odds if outcome in ("win", "place") else 0.0
        return win_return + place_return

    def _clv_pct(self, odds_taken: float, closing_odds: float | None) -> float | None:
        """Closing-line value as a % price improvement: positive means the price
        taken beat the closing (Betfair SP) price. None when no closing price."""
        if not closing_odds or closing_odds <= 1.0 or not odds_taken:
            return None
        return round((odds_taken / closing_odds - 1.0) * 100.0, 2)

    def settle_bet(self, bet_id: int, outcome: str, closing_odds: float | None = None) -> dict:
        """
        Record the outcome of a pending bet and credit the gross return to the bankroll.

        outcome: 'win' | 'place' | 'lose' | 'void' ('void' refunds the stake).
        closing_odds: the closing (Betfair SP) price, used to compute and store CLV.
        Returns the fully-populated bet row as a dict.
        """
        outcome = outcome.lower()
        if outcome not in ("win", "place", "lose", "void"):
            raise ValueError(f"outcome must be win/place/lose/void, got '{outcome}'")
        row = self._conn().execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()
        if row is None:
            raise ValueError(f"bet {bet_id} not found")
        if row["outcome"] is not None:
            raise ValueError(f"bet {bet_id} already settled as '{row['outcome']}'")

        if outcome == "void":
            gross_return = row["stake"]  # full stake refund
        else:
            gross_return = self._calc_gross_return(
                row["bet_type"], row["stake"], row["odds_decimal"], outcome
            )
        profit = gross_return - row["stake"]
        clv_pct = self._clv_pct(row["odds_decimal"], closing_odds)
        settled_at = now().isoformat()
        new_balance = self.bankroll + gross_return

        with self._pool.write_lock():
            conn = self._conn()
            conn.execute(
                "UPDATE bets SET outcome=?, settled_at=?, gross_return=?, profit=?, "
                "closing_odds=?, clv_pct=? WHERE id=?",
                (outcome, settled_at, round(gross_return, 4), round(profit, 4),
                 closing_odds, clv_pct, bet_id),
            )
            conn.execute(
                "INSERT INTO bankroll_log(logged_at, event_type, bet_id, amount, balance) "
                "VALUES (?,?,?,?,?)",
                (settled_at, "bet_settled", bet_id, gross_return, new_balance),
            )
            conn.commit()
        logger.info(
            "bet_tracker: settled #%d %s outcome=%s profit=%.2f bankroll=%.2f",
            bet_id, row["horse_name"], outcome, profit, new_balance,
        )
        get_notifier().notify_bet_outcome(row["horse_name"], outcome, round(profit, 2), new_balance)
        return dict(self._conn().execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_bets(self) -> list[dict]:
        return [dict(r) for r in self._conn().execute(
            "SELECT * FROM bets WHERE outcome IS NULL ORDER BY placed_at"
        ).fetchall()]

    def all_bets(self, settled_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM bets"
        if settled_only:
            sql += " WHERE outcome IS NOT NULL"
        sql += " ORDER BY placed_at"
        return [dict(r) for r in self._conn().execute(sql).fetchall()]

    def get_bet(self, bet_id: int) -> dict | None:
        row = self._conn().execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Dashboard-ready aggregations
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Overall performance snapshot keyed for dashboard consumption."""
        df = pd.DataFrame(self.all_bets(settled_only=True))
        if df.empty:
            return {
                "total_bets": 0,
                "pending_bets": len(self.pending_bets()),
                "wins": 0,
                "places": 0,
                "win_rate": 0.0,
                "place_rate": 0.0,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi_pct": 0.0,
                "bankroll": round(self.bankroll, 2),
                "initial_bankroll": round(self.initial_bankroll, 2),
                "max_drawdown_pct": 0.0,
                "avg_clv_pct": None,
                "stop_loss_active": self.is_stopped(),
            }
        total = len(df)
        wins = int((df["outcome"] == "win").sum())
        places = int((df["outcome"] == "place").sum())
        staked = float(df["stake"].sum())
        profit = float(df["profit"].sum())
        roi = (profit / staked * 100.0) if staked > 0 else 0.0

        clv = df["clv_pct"].dropna() if "clv_pct" in df.columns else pd.Series(dtype=float)
        avg_clv = round(float(clv.mean()), 2) if not clv.empty else None

        log_rows = self._conn().execute(
            "SELECT balance FROM bankroll_log ORDER BY id"
        ).fetchall()
        max_dd = 0.0
        if log_rows:
            peak = float(log_rows[0]["balance"])
            for r in log_rows:
                b = float(r["balance"])
                peak = max(peak, b)
                if peak > 0:
                    max_dd = max(max_dd, (peak - b) / peak)

        return {
            "total_bets": total,
            "pending_bets": len(self.pending_bets()),
            # Raw counts travel with the rates so every surface can show the
            # denominator (realized win rate = wins/total, requirement 9).
            "wins": wins,
            "places": places,
            "win_rate": round(wins / total, 4),
            "place_rate": round((wins + places) / total, 4),
            "total_staked": round(staked, 2),
            "total_profit": round(profit, 2),
            "roi_pct": round(roi, 2),
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_clv_pct": avg_clv,
            "stop_loss_active": self.is_stopped(),
        }

    def pl_series(self) -> pd.DataFrame:
        """Settled bets with a running cumulative P&L column — ready for a line chart."""
        df = pd.DataFrame(self.all_bets(settled_only=True))
        if df.empty:
            return pd.DataFrame(
                columns=["settled_at", "horse_name", "venue", "odds_decimal",
                         "stake", "outcome", "profit", "cumulative_profit"]
            )
        cols = ["settled_at", "horse_name", "venue", "odds_decimal", "stake", "outcome", "profit"]
        df = df[cols].sort_values("settled_at").reset_index(drop=True)
        df["cumulative_profit"] = df["profit"].cumsum()
        return df

    def breakdown(self, by: str = "venue") -> pd.DataFrame:
        """P&L breakdown grouped by a column (venue / strategy / bet_type / outcome)."""
        df = pd.DataFrame(self.all_bets(settled_only=True))
        if df.empty or by not in df.columns:
            return pd.DataFrame()
        g = (
            df.groupby(by)
            .agg(
                bets=("id", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                places=("outcome", lambda x: (x == "place").sum()),
                total_staked=("stake", "sum"),
                total_profit=("profit", "sum"),
            )
            .reset_index()
        )
        g["win_rate"] = (g["wins"] / g["bets"]).round(4)
        g["roi_pct"] = ((g["total_profit"] / g["total_staked"]) * 100).round(2)
        return g.sort_values("total_profit", ascending=False).reset_index(drop=True)

    def bankroll_history(self) -> pd.DataFrame:
        """Full bankroll_log as a DataFrame for equity-curve charts."""
        rows = [dict(r) for r in self._conn().execute(
            "SELECT logged_at, event_type, amount, balance FROM bankroll_log ORDER BY id"
        ).fetchall()]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["logged_at", "event_type", "amount", "balance"]
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: str | Path | None = None) -> Path:
        """Write all bets to CSV. Returns the path written."""
        out = Path(path or _DEFAULT_CSV)
        out.parent.mkdir(parents=True, exist_ok=True)
        bets = self.all_bets()
        if not bets:
            out.write_text("no bets recorded\n", encoding="utf-8")
            logger.warning("bet_tracker: no bets to export, wrote placeholder to %s", out)
            return out
        pd.DataFrame(bets).to_csv(out, index=False)
        logger.info("bet_tracker: exported %d bets to %s", len(bets), out)
        return out

"""Settings page — MASHR’s Predictor v6.

Editable sections: proxy pool, scraping/betting thresholds, model training config
(including version tag), and drift-retrain thresholds. Validates inputs before
saving and writes atomically to config.yaml. A "Reload Services" panel clears
all Streamlit caches so the next page action picks up the new config.

Launch:
    streamlit run ui/settings.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st
import yaml

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CFG_PATH = _ROOT / "config.yaml"

# Pattern that matches http(s)://[user:pass@]host:port  — basic sanity check.
_PROXY_RE = re.compile(
    r"^https?://(?:[^@]+@)?[a-zA-Z0-9._\-]+(:\d{1,5})?$"
)
# Valid characters for a version tag (alphanumeric, hyphen, dot, underscore).
_VTAG_RE = re.compile(r"^[a-zA-Z0-9._\-]{1,32}$")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Settings — MASHR’s Predictor v6",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── design system ────────────────────────────────────────────
# A single-column form page, so it takes the narrow shell. Dark, like the rest
# of the app — this module used to ship its own 73-line light stylesheet with a
# duplicate :root palette; its classes now live in ui/_design.py.
from ui._design import inject_design  # noqa: E402

inject_design(width="narrow")


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=0)
def _load_cfg() -> dict:
    try:
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return {}


def _save_cfg(cfg: dict) -> None:
    """Atomic write: temp file → replace so a crash mid-write leaves config intact."""
    tmp = _CFG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    tmp.replace(_CFG_PATH)
    _load_cfg.clear()


def _validate_proxies(text: str) -> tuple[list[str], list[str]]:
    """Return (valid_urls, invalid_lines). Blank lines are skipped."""
    valid, invalid = [], []
    seen: set[str] = set()
    for line in text.splitlines():
        url = line.strip()
        if not url:
            continue
        if url in seen:
            continue  # silent dedup
        seen.add(url)
        if _PROXY_RE.match(url):
            valid.append(url)
        else:
            invalid.append(url)
    return valid, invalid


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.caption(
        "Changes take effect after Save. Scrapers and the model trainer read "
        "config.yaml fresh on each run — no process restart needed."
    )
    st.markdown("---")
    st.markdown("### Navigation")
    # These pages are standalone Streamlit scripts (not a pages/ multipage app),
    # so st.page_link would raise. List the launch commands instead.
    st.caption(
        "Each page is launched on its own, e.g.\n\n"
        "`streamlit run ui/app.py`\n\n"
        "Available: app · live_races · predictions · bet_placer · "
        "performance_dashboard · race_compare"
    )

# ── page header ───────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Settings")
st.caption("Configure proxy pool, betting/scraping thresholds, and model training parameters.")

cfg = _load_cfg()

# ── save-state banners (shown after successful save / validation errors) ───────
if st.session_state.get("_settings_saved"):
    st.markdown(
        '<div class="banner-ok">✓ Configuration saved to config.yaml</div>',
        unsafe_allow_html=True,
    )
    del st.session_state["_settings_saved"]

for err in st.session_state.pop("_settings_errors", []):
    st.markdown(f'<div class="banner-err">✗ {err}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FORM — all edits batched; only written to disk on submit after validation
# ══════════════════════════════════════════════════════════════════════════════
with st.form("settings_form"):

    # ── 1 · Proxy Pool ────────────────────────────────────────────────────────
    st.markdown('<div class="settings-card"><h4>Proxy Pool</h4>', unsafe_allow_html=True)

    proxy_cfg = cfg.get("proxy_pool", {})
    proxy_enabled = st.toggle(
        "Enable proxy rotation",
        value=bool(proxy_cfg.get("enabled", False)),
        help="When enabled, each HTTP request cycles through the proxy list below.",
    )
    raw_proxies: list[str] = proxy_cfg.get("proxies") or []
    proxy_text = st.text_area(
        "Proxy URLs (one per line)",
        value="\n".join(raw_proxies),
        height=110,
        help=(
            "Format: http://LOGIN:PASSWORD@HOST:PORT  "
            "Supports http:// and https://. "
            "DataImpulse residential proxies work well for Irish bookmakers."
        ),
        placeholder="http://user:pass@gw.provider.com:8080",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2 · Scraping & Betting Thresholds ─────────────────────────────────────
    st.markdown(
        '<div class="settings-card"><h4>Scraping &amp; Betting Thresholds</h4>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        scrape_interval = st.number_input(
            "Scrape interval (seconds)",
            min_value=60,
            max_value=86400,
            value=int(cfg.get("scrape_interval", 3600)),
            step=60,
            help="How often scrapers run. Also used as the cache TTL for paddy_power.json.",
        )
        each_way_threshold = st.number_input(
            "Each-way threshold (decimal odds)",
            min_value=1.0,
            max_value=100.0,
            value=float(cfg.get("each_way_threshold", 8.0)),
            step=0.5,
            format="%.1f",
            help="Minimum decimal odds before the E/W value badge is shown on a selection.",
        )
    with col2:
        low_odds_threshold = st.number_input(
            "Low odds threshold (decimal odds)",
            min_value=1.0,
            max_value=10.0,
            value=float(cfg.get("low_odds_threshold", 2.0)),
            step=0.1,
            format="%.1f",
            help="Runners at or below this odds are flagged is_low_odds (odds-on favourites).",
        )
        gbp_eur_rate = st.number_input(
            "GBP → EUR rate",
            min_value=0.5,
            max_value=2.5,
            value=float(cfg.get("gbp_eur_rate", 1.18)),
            step=0.01,
            format="%.2f",
            help="Static exchange rate used for GBP→EUR display tooltips in the UI.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3 · Model Training ────────────────────────────────────────────────────
    st.markdown('<div class="settings-card"><h4>Model Training</h4>', unsafe_allow_html=True)

    model_cfg = cfg.get("model", {})
    col3, col4 = st.columns(2)
    with col3:
        version_tag = st.text_input(
            "Model version tag",
            value=str(model_cfg.get("version_tag", "v3")),
            max_chars=32,
            help=(
                "Controls the filename of saved artefacts: catboost_{target}_{tag}.bin "
                "and catboost_{tag}_meta.json. Change to create a new versioned set "
                "without overwriting the current models. Allowed chars: a-z A-Z 0-9 . - _"
            ),
            placeholder="v3",
        )
        test_size = st.slider(
            "Hold-out test size",
            min_value=0.05,
            max_value=0.40,
            value=float(model_cfg.get("test_size", 0.2)),
            step=0.05,
            format="%.0f%%",
            help="Fraction of races reserved for time-based hold-out evaluation.",
        )
        cv_folds = st.number_input(
            "CV folds",
            min_value=2,
            max_value=10,
            value=int(model_cfg.get("cv_folds", 5)),
            step=1,
            help="Stratified k-fold count used during Optuna hyperparameter search.",
        )
        optuna_trials = st.number_input(
            "Optuna trials",
            min_value=1,
            max_value=500,
            value=int(model_cfg.get("optuna_trials", 50)),
            step=5,
            help="Number of Optuna TPE trials per target. Higher = slower but better hyperparams.",
        )
    with col4:
        iterations = st.number_input(
            "CatBoost iterations",
            min_value=100,
            max_value=10000,
            value=int(model_cfg.get("iterations", 1000)),
            step=100,
            help="Maximum boosting rounds per CatBoost model.",
        )
        early_stopping_rounds = st.number_input(
            "Early stopping rounds",
            min_value=5,
            max_value=500,
            value=int(model_cfg.get("early_stopping_rounds", 50)),
            step=5,
            help="Training halts if validation AUC hasn't improved for this many rounds.",
        )
        show_positions = st.number_input(
            "Show positions (top-N)",
            min_value=1,
            max_value=6,
            value=int(model_cfg.get("show_positions", 3)),
            step=1,
            help="Finishing position ≤ this value counts as 'showed' for the third target.",
        )
        max_sample_weight = st.number_input(
            "Max sample weight",
            min_value=1,
            max_value=100,
            value=int(model_cfg.get("max_sample_weight", 20)),
            step=1,
            help=(
                "Odds-inverse weights (1/implied_prob) are clipped to this ceiling — "
                "prevents extreme oversampling of longshots."
            ),
        )
    all_targets = ["won", "placed_2", "showed"]
    targets = st.multiselect(
        "Training targets",
        options=all_targets,
        default=model_cfg.get("targets", all_targets),
        help="Which binary classifiers to train. Deselecting a target skips that model file.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 4 · Drift & Retrain ───────────────────────────────────────────────────
    st.markdown(
        '<div class="settings-card"><h4>Drift Detection &amp; Retrain</h4>',
        unsafe_allow_html=True,
    )
    retrain_cfg = cfg.get("retrain", {})
    col5, col6 = st.columns(2)
    with col5:
        psi_threshold = st.number_input(
            "PSI threshold",
            min_value=0.01,
            max_value=1.0,
            value=float(retrain_cfg.get("psi_threshold", 0.2)),
            step=0.01,
            format="%.2f",
            help="PSI > this on any feature flags major drift (>0.2 = major shift).",
        )
        ks_pvalue_threshold = st.number_input(
            "KS p-value threshold",
            min_value=0.001,
            max_value=0.5,
            value=float(retrain_cfg.get("ks_pvalue_threshold", 0.05)),
            step=0.005,
            format="%.3f",
            help="KS two-sample p-value < this also flags drift. Lower = stricter.",
        )
        min_drift_features = st.number_input(
            "Min drift features",
            min_value=1,
            max_value=10,
            value=int(retrain_cfg.get("min_drift_features", 1)),
            step=1,
            help="How many features must show drift before retraining is triggered.",
        )
    with col6:
        archive_keep = st.number_input(
            "Archive keep (versions)",
            min_value=1,
            max_value=20,
            value=int(retrain_cfg.get("archive_keep", 5)),
            step=1,
            help="Past model versions retained in models/archive/ before pruning.",
        )
        auc_decay_threshold = st.number_input(
            "AUC decay threshold",
            min_value=0.0,
            max_value=0.5,
            value=float(retrain_cfg.get("auc_decay_threshold", 0.05)),
            step=0.01,
            format="%.2f",
            help="Warn if new model AUC drops more than this fraction vs previous run.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── submit ────────────────────────────────────────────────────────────────
    submitted = st.form_submit_button(
        "💾  Save Configuration",
        type="primary",
        width="stretch",
    )

# ── validation & save (outside form so errors render above it on rerun) ────────
if submitted:
    errors: list[str] = []

    # proxy validation
    valid_proxies, bad_proxies = _validate_proxies(proxy_text)
    for bp in bad_proxies:
        errors.append(f"Invalid proxy URL (must start with http:// or https://): {bp!r}")

    if proxy_enabled and not valid_proxies:
        errors.append(
            "Proxy rotation is enabled but no valid proxy URLs were supplied. "
            "Add at least one URL or disable proxy rotation."
        )

    # version tag validation
    vtag = version_tag.strip()
    if not vtag:
        errors.append("Model version tag cannot be blank.")
    elif not _VTAG_RE.match(vtag):
        errors.append(
            f"Model version tag {vtag!r} contains invalid characters. "
            "Use only letters, digits, hyphens, underscores, or dots (max 32 chars)."
        )

    # targets validation
    if not targets:
        errors.append("Select at least one training target (won / placed_2 / showed).")

    if errors:
        st.session_state["_settings_errors"] = errors
        st.rerun()
    else:
        cfg["proxy_pool"] = {"enabled": proxy_enabled, "proxies": valid_proxies}
        cfg["scrape_interval"] = int(scrape_interval)
        cfg["each_way_threshold"] = float(each_way_threshold)
        cfg["low_odds_threshold"] = float(low_odds_threshold)
        cfg["gbp_eur_rate"] = float(round(gbp_eur_rate, 4))

        cfg["model"] = {
            **cfg.get("model", {}),
            "version_tag": vtag,
            "test_size": float(test_size),
            "cv_folds": int(cv_folds),
            "optuna_trials": int(optuna_trials),
            "early_stopping_rounds": int(early_stopping_rounds),
            "iterations": int(iterations),
            "show_positions": int(show_positions),
            "max_sample_weight": int(max_sample_weight),
            "targets": list(targets),
        }

        cfg["retrain"] = {
            **cfg.get("retrain", {}),
            "psi_threshold": float(round(psi_threshold, 4)),
            "ks_pvalue_threshold": float(round(ks_pvalue_threshold, 4)),
            "min_drift_features": int(min_drift_features),
            "archive_keep": int(archive_keep),
            "auc_decay_threshold": float(round(auc_decay_threshold, 4)),
        }

        _save_cfg(cfg)
        st.session_state["_settings_saved"] = True
        st.rerun()

# ── Reload Services panel ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Reload Services")
st.markdown(
    '<div class="reload-card">',
    unsafe_allow_html=True,
)
st.caption(
    "Scrapers and the model trainer read config.yaml directly on each CLI invocation "
    "so no reload is needed for them. The buttons below clear Streamlit in-memory "
    "caches so dashboard pages reflect the new config immediately without restarting."
)
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔄  Clear all page caches", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("All Streamlit data and resource caches cleared.")

with col_b:
    # Trigger a fresh prediction run using the updated config (new version_tag etc.)
    if st.button("🎯  Re-run predictions now", width="stretch"):
        try:
            from models.predictor import Predictor
            p = Predictor()
            loaded = p.load()
            if loaded:
                races = p.predict()
                st.cache_data.clear()
                st.success(f"Predictions refreshed — {len(races)} races scored.")
            else:
                st.warning(
                    "No model files found for the current version tag. "
                    "Run `python -m models.train` to train models first."
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction run failed: {exc}")

st.markdown("</div>", unsafe_allow_html=True)

# ── config snapshot (read-only, for reference) ────────────────────────────────
with st.expander("View raw config.yaml (read-only)"):
    live_cfg = _load_cfg()
    st.code(yaml.dump(live_cfg, default_flow_style=False, sort_keys=False), language="yaml")

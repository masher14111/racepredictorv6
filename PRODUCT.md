# Product

## Register

product

## Users

A single power-user (the owner/operator) who checks the dashboard on race day before
placing **real money** bets. They are numerate, understand racing and betting markets,
and already trust the underlying modelling — what they need from the screen is a fast,
honest read on _who to back and why_. Context of use: at a desk, often under time
pressure between races, deciding stake and selection in the minutes before off-time.
Not a novice audience; no hand-holding required, but the numbers must be legible and
trustworthy at a glance.

## Product Purpose

Race Predictor v3 turns a horse-racing ML pipeline (scrapers → features → calibrated
CatBoost models → value layer) into actionable betting decisions. The dashboard exists
to answer one question per race: **is there a bet here, and on whom?**

Success is not "show predictions" — it's surfacing genuine **value** (edge vs. the
market) using a price-free, calibrated win probability, gated by an odds band so both
sensible favourites and qualified longshots can be flagged. The interface must make
calibrated confidence, expected value, and each-way viability immediately readable, and
must be honest when data is stale, thin, or low-confidence rather than papering over it.

## Brand Personality

**Premium · confident · refined.** Reads like a private, high-end betting desk, not a
high-street bookmaker. Voice is calm, precise, and numbers-forward — no hype, no urgency
banners, no exclamation. Confidence is communicated through restraint and the quality of
the data presentation, not through loud color or motion. The owner should feel they are
using a sharp professional instrument.

## Anti-references

- **Flashy gambling / bookmaker sites** — no neon, gold-on-black, casino sparkle, "BET
  NOW" urgency, countdown pressure, or hype copy. The product is analytical, not
  promotional.
- **Generic AI SaaS template** — no cream/sand body backgrounds, no per-section uppercase
  eyebrow kickers, no identical icon-heading-text card grids, no hero-metric splash, no
  gradient text. Density and craft over template scaffolding.

## Design Principles

1. **Trust the numbers.** Probabilities shown are calibrated; value is computed from a
   price-free model to avoid market circularity. The UI's job is to present that honesty
   faithfully — never inflate, never imply more certainty than the model has.
2. **Value over favourites.** Lead with edge / expected value, not raw win%. A low-odds
   favourite and a gated longshot can both be "the bet"; the layout must let either rise
   to the top on merit.
3. **Decision-ready at a glance.** One scan per race card should answer who, how
   confident, what edge, and whether each-way applies. Built for speed on race day —
   density is a feature, not a flaw, as long as hierarchy stays clean.
4. **Quiet confidence.** Premium restraint: a disciplined palette, tabular numerics,
   considered spacing. The design earns trust by looking like it knows what it's doing.
5. **Honest about state.** Stale caches, missing odds, thin fields, and below-threshold
   selections are surfaced plainly, not hidden. The user must always know how much to
   trust what's on screen.

## Accessibility & Inclusion

- **WCAG AA contrast.** Body text ≥4.5:1 against its background; large/bold text ≥3:1.
  No muted-gray-on-tinted-white. Numerics (odds, probabilities) must stay crisp.
- **Reduced-motion safe.** Any motion ships with a `prefers-reduced-motion: reduce`
  fallback (crossfade or instant); no essential content gated behind animation.
- Color is never the sole carrier of meaning (rank, value, each-way, status): pair with
  text, icon, or shape so the screen reads under varied lighting and color-vision needs.

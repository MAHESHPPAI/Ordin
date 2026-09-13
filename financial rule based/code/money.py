"""
money.py — Decimal-safe monetary helpers and FX conversion.

All monetary arithmetic uses ``decimal.Decimal`` at full internal precision.
Rounding to 2 decimal places (ROUND_HALF_UP) happens *only* at output time.

FX lookup strategy (§ get_rate):
  1. Exact (from, to, date) match in exchange_rates.csv.
  2. Inverse pair (to, from, date) → use 1 / rate.
  3. Nearest *prior* rate_date for the same pair (or its inverse).
  4. If none found, raise an error — never silently guess.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import EXCHANGE_RATES_CSV

logger = logging.getLogger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────────
_PairDate = Tuple[str, str, date]            # (from, to, date)
_PairKey  = Tuple[str, str]                  # (from, to)

# ── Module-level FX cache ────────────────────────────────────────────────────
_rate_cache: Dict[_PairDate, Decimal] = {}
# Sorted list of dates per currency pair for nearest-prior lookup
_pair_dates: Dict[_PairKey, list[date]] = {}
_loaded: bool = False


def _parse_date(s: str) -> date:
    """Parse a YYYY-MM-DD string into a ``datetime.date``."""
    return date.fromisoformat(s)


def load_rates(path: Path = EXCHANGE_RATES_CSV) -> None:
    """Read exchange_rates.csv into the module-level caches."""
    global _loaded
    _rate_cache.clear()
    _pair_dates.clear()

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rd = _parse_date(row["rate_date"])
            fc = row["from_currency"].strip()
            tc = row["to_currency"].strip()
            rate = Decimal(row["rate"].strip())
            key: _PairDate = (fc, tc, rd)
            _rate_cache[key] = rate
            pair: _PairKey = (fc, tc)
            _pair_dates.setdefault(pair, []).append(rd)

    # Sort date lists once for binary search later
    for pair in _pair_dates:
        _pair_dates[pair].sort()

    _loaded = True
    logger.info("Loaded %d FX rate rows covering %d pair(s).",
                len(_rate_cache), len(_pair_dates))


def _ensure_loaded() -> None:
    if not _loaded:
        load_rates()


def _nearest_prior_date(pair: _PairKey, target: date) -> Optional[date]:
    """Return the latest date ≤ ``target`` for ``pair``, or ``None``."""
    dates = _pair_dates.get(pair)
    if not dates:
        return None
    # Simple linear scan (only ~40 dates per pair at most)
    best: Optional[date] = None
    for d in dates:
        if d <= target:
            best = d
        else:
            break  # sorted, so nothing later can be ≤ target
    return best


def get_rate(from_currency: str, to_currency: str, on_date: date) -> Decimal:
    """
    Look up a conversion rate for *from_currency* → *to_currency* on *on_date*.

    Strategy:
      1. Exact match (from, to, date).
      2. Inverse pair (to, from, date)  →  1 / rate.
      3. Nearest prior date for either direction.
      4. Raise ``ValueError`` if nothing found.
    """
    _ensure_loaded()

    if from_currency == to_currency:
        return Decimal("1")

    # 1. Exact match
    exact = _rate_cache.get((from_currency, to_currency, on_date))
    if exact is not None:
        return exact

    # 2. Inverse exact
    inv_exact = _rate_cache.get((to_currency, from_currency, on_date))
    if inv_exact is not None:
        return Decimal("1") / inv_exact

    # 3a. Nearest prior — direct pair
    fwd_pair: _PairKey = (from_currency, to_currency)
    fwd_date = _nearest_prior_date(fwd_pair, on_date)
    if fwd_date is not None:
        rate = _rate_cache[(from_currency, to_currency, fwd_date)]
        logger.warning(
            "FX fallback: used %s rate from %s instead of %s for %s→%s",
            fwd_date.isoformat(), fwd_date.isoformat(), on_date.isoformat(),
            from_currency, to_currency,
        )
        return rate

    # 3b. Nearest prior — inverse pair
    inv_pair: _PairKey = (to_currency, from_currency)
    inv_date = _nearest_prior_date(inv_pair, on_date)
    if inv_date is not None:
        inv_rate = _rate_cache[(to_currency, from_currency, inv_date)]
        logger.warning(
            "FX fallback (inverse): used %s rate from %s instead of %s "
            "for %s→%s (inverted %s→%s)",
            inv_date.isoformat(), inv_date.isoformat(), on_date.isoformat(),
            from_currency, to_currency, to_currency, from_currency,
        )
        return Decimal("1") / inv_rate

    raise ValueError(
        f"No FX rate found for {from_currency}→{to_currency} "
        f"on or before {on_date.isoformat()}"
    )


def convert_to_home_currency(
    amount: Decimal,
    from_currency: str,
    home_currency: str,
    settlement_date: date,
) -> Decimal:
    """Convert *amount* in *from_currency* to *home_currency* at full precision."""
    if from_currency == home_currency:
        return amount
    rate = get_rate(from_currency, home_currency, settlement_date)
    return amount * rate


def round_money(value: Decimal) -> Decimal:
    """Round to 2 decimal places using ROUND_HALF_UP — call only at output time."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def to_decimal(value: str) -> Optional[Decimal]:
    """
    Safely convert a string to Decimal.

    Returns ``None`` for blank / whitespace-only strings.
    Raises ``InvalidOperation`` for genuinely malformed values.
    """
    v = value.strip() if value else ""
    if not v:
        return None
    return Decimal(v)

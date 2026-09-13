"""
safe_amount.py — Stage 4: Safe Amount and Full-Payment Timing Engine.

Calculates:
1. `amount_safe_to_pay`: The maximum amount the user can safely pay today
   (on request_date) before optional spending changes, while maintaining
   minimum_balance_to_keep across the entire 90-day forecast.
   Enforces: 0 <= amount_safe_to_pay <= requested_amount.
2. `earliest_date_for_full_payment`: The earliest date within the 90-day window
   when paying requested_amount in full as a single lump-sum payment is forecast
   to be completely safe without optional spending changes.
   Returns request_date if safe immediately, a future date if safe later, or None.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import config
from canonical import CanonicalEvent
from money import round_money
from simulator import SimulationResult, simulate_timeline

logger = logging.getLogger(__name__)


def compute_amount_safe_to_pay(
    request: Dict[str, Any],
    forecast_events: List[CanonicalEvent],
    user_profile: Dict[str, Any],
    forecast_days: int = config.FORECAST_DAYS,
) -> Decimal:
    """
    Computes the largest amount the user can safely pay on request_date
    before optional spending changes, capped at requested_amount.

    Formula:
        min_headroom = min_{t} (balance(t) - minimum_balance_to_keep)
        amount_safe_to_pay = max(0, min(requested_amount, min_headroom))
    """
    req_date: date = request["request_date"]
    req_amount: Decimal = request["requested_amount"]
    start_bal: Decimal = user_profile["current_available_balance"]
    min_bal: Decimal = user_profile["minimum_balance_to_keep"]

    # Baseline 90-day simulation without proposed payments or spending changes
    sim: SimulationResult = simulate_timeline(
        starting_balance=start_bal,
        minimum_balance_to_keep=min_bal,
        forecast_events=forecast_events,
        request_date=req_date,
        forecast_days=forecast_days,
        proposed_payments=None,
        spending_adjustments=None,
    )

    if sim.min_headroom <= Decimal("0"):
        return Decimal("0")

    safe_amount = min(req_amount, sim.min_headroom)
    return round_money(max(Decimal("0"), safe_amount))


def find_earliest_date_for_full_payment(
    request: Dict[str, Any],
    forecast_events: List[CanonicalEvent],
    user_profile: Dict[str, Any],
    forecast_days: int = config.FORECAST_DAYS,
) -> Optional[date]:
    """
    Finds the earliest date in [request_date, request_date + forecast_days]
    where paying requested_amount in full as a single payment passes the
    90-day safety check (without optional spending changes).

    Returns:
        date if safe on or after request_date within 90 days; None otherwise.
    """
    req_date: date = request["request_date"]
    req_amount: Decimal = request["requested_amount"]
    start_bal: Decimal = user_profile["current_available_balance"]
    min_bal: Decimal = user_profile["minimum_balance_to_keep"]

    # Check if full payment is safe today
    safe_today = compute_amount_safe_to_pay(
        request=request,
        forecast_events=forecast_events,
        user_profile=user_profile,
        forecast_days=forecast_days,
    )
    if safe_today == req_amount:
        return req_date

    # Evaluate future candidate dates day-by-day
    for day_offset in range(1, forecast_days + 1):
        cand_date = req_date + timedelta(days=day_offset)
        sim = simulate_timeline(
            starting_balance=start_bal,
            minimum_balance_to_keep=min_bal,
            forecast_events=forecast_events,
            request_date=req_date,
            forecast_days=forecast_days,
            proposed_payments=[(cand_date, req_amount)],
            spending_adjustments=None,
        )
        if sim.is_safe:
            return cand_date

    return None

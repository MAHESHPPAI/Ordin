"""
simulator.py — Stage 4: 90-Day Cash Balance Walk Simulator.

Simulates the daily available balance trajectory across the 90-day forecast window
starting from authoritative current_available_balance as of request_date.

Key Responsibilities:
1. Performs daily balance walk for days 0 to 90 (request_date to request_date + FORECAST_DAYS).
2. Starting balance is strictly current_available_balance from financial_profiles.csv.
3. Incorporates baseline forecasted cash flows (credits positive, debits negative).
4. Supports optional proposed payment schedules (e.g. installments, single payments).
5. Supports optional spending changes (stopping or reducing flexible recurring events).
6. Computes daily balances, minimum balance reached, minimum headroom above
   minimum_balance_to_keep, and safety flags.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import config
from canonical import CanonicalEvent
from money import round_money

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a 90-day balance walk simulation."""
    is_safe: bool                           # True if balance >= minimum_balance_to_keep every day
    starting_balance: Decimal               # balance as of request_date
    minimum_balance_to_keep: Decimal        # required reserve floor
    min_balance: Decimal                    # lowest balance reached during 90 days
    min_headroom: Decimal                   # min_balance - minimum_balance_to_keep
    min_balance_date: date                  # date when lowest balance occurred
    violation_date: Optional[date]          # first date balance < minimum_balance_to_keep, if any
    daily_balances: Dict[date, Decimal]     # balance at end of each day
    daily_cash_flows: Dict[date, Decimal]   # net cash flow on each day
    days_simulated: int = config.FORECAST_DAYS


def simulate_timeline(
    starting_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    forecast_events: List[CanonicalEvent],
    request_date: date,
    forecast_days: int = config.FORECAST_DAYS,
    proposed_payments: Optional[List[Tuple[date, Decimal]]] = None,
    spending_adjustments: Optional[Dict[str, Optional[Decimal]]] = None,
) -> SimulationResult:
    """
    Simulates daily balance for 90 days starting from request_date.

    Parameters:
        starting_balance: current_available_balance as of request_date
        minimum_balance_to_keep: required safety buffer floor
        forecast_events: list of CanonicalEvent objects in the forecast window
        request_date: anchor date of the evaluation request
        forecast_days: number of days to simulate (default 90)
        proposed_payments: optional list of (payment_date, debit_amount) for candidate purchase
        spending_adjustments: optional dict of {event_id_or_category: new_amount_or_none}
                              where None means stopped (0), and Decimal means reduced to amount.

    Returns:
        SimulationResult containing balance trajectory, headroom, and safety status.
    """
    # 1. Aggregate daily net cash flows
    daily_cash_flows: Dict[date, Decimal] = defaultdict(Decimal)

    adjustments = spending_adjustments or {}

    for ev in forecast_events:
        if not ev.included_in_forecast or ev.amount is None:
            continue

        ev_date = ev.settlement_date or ev.event_date
        if ev_date is None:
            continue

        effective_amount = ev.amount

        # Check if a spending adjustment modifies this event (by event_id or category)
        if ev.event_id in adjustments:
            adj = adjustments[ev.event_id]
            effective_amount = Decimal("0") if adj is None else adj
        elif ev.category in adjustments:
            adj = adjustments[ev.category]
            effective_amount = Decimal("0") if adj is None else adj

        # Direction: credit adds, debit subtracts
        if ev.direction == "credit":
            daily_cash_flows[ev_date] += effective_amount
        else:
            daily_cash_flows[ev_date] -= effective_amount

    # 2. Add proposed purchase payments (debits)
    if proposed_payments:
        for p_date, p_amount in proposed_payments:
            daily_cash_flows[p_date] -= p_amount

    # 3. Simulate day-by-day balance walk
    cur_balance = starting_balance
    daily_balances: Dict[date, Decimal] = {}

    min_balance = cur_balance
    min_headroom = cur_balance - minimum_balance_to_keep
    min_balance_date = request_date
    violation_date: Optional[date] = None

    # Day 0 is request_date (including any net cash flow on request_date)
    for day_offset in range(0, forecast_days + 1):
        cur_date = request_date + timedelta(days=day_offset)

        # Apply net cash flow on this day
        if cur_date in daily_cash_flows:
            cur_balance += daily_cash_flows[cur_date]
        daily_balances[cur_date] = cur_balance
        headroom = cur_balance - minimum_balance_to_keep

        if headroom < min_headroom:
            min_headroom = headroom
            min_balance = cur_balance
            min_balance_date = cur_date

        if cur_balance < minimum_balance_to_keep and violation_date is None:
            violation_date = cur_date

    is_safe = min_headroom >= Decimal("0")

    return SimulationResult(
        is_safe=is_safe,
        starting_balance=starting_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        min_balance=round_money(min_balance),
        min_headroom=round_money(min_headroom),
        min_balance_date=min_balance_date,
        violation_date=violation_date,
        daily_balances=daily_balances,
        daily_cash_flows=dict(daily_cash_flows),
        days_simulated=forecast_days,
    )

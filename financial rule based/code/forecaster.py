"""
forecaster.py — Stage 3: Cash-Flow Timeline Forecaster.

Projects each recurring series forward across the 90-day forecast window
(request_date to request_date + FORECAST_DAYS) per request.

Core Responsibilities:
1. Projects recurring series based on inferred RecurrencePatterns from recurrence.py.
2. Splices in explicit scheduled/pending-debit rows from Stage 1 output and Stage 2
   message-confirmed cash events.
3. Deduplicates projections against explicit facts within a date window (+/- 4 days)
   so no cash event is double-counted.
4. Respects Stage 2 suppression signals (e.g. 'employment ended' halts salary projections).
5. Carries direction, category, flexibility, minimum_allowed_amount forward unchanged.
6. Emits CanonicalEvents with `projected=True` for forecasts and `projected=False`
   for confirmed/explicit facts.
7. Validates defensive invariant: asserts 0 overlap between protected and flexible
   categories at the user profile level.

DOES NOT perform balance walks, safety checks, or policy evaluations (strictly Stage 4).
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from canonical import CanonicalEvent
from money import round_money
from recurrence import RecurrencePattern, detect_user_recurrence_patterns

logger = logging.getLogger(__name__)

# Deduplication window: if a projected event is within this many days of an
# explicit event of the same user, category, and direction, the explicit event wins.
DEDUPE_WINDOW_DAYS: int = 4


def add_months_clamped(orig_date: date, months_to_add: int, target_day: int) -> date:
    """
    Advances a date by months_to_add, setting day to target_day clamped
    to the maximum days in the resulting month.
    """
    new_year = orig_date.year + (orig_date.month + months_to_add - 1) // 12
    new_month = (orig_date.month + months_to_add - 1) % 12 + 1
    max_days = calendar.monthrange(new_year, new_month)[1]
    clamped_day = min(target_day, max_days)
    return date(new_year, new_month, clamped_day)


def generate_candidate_dates(
    pat: RecurrencePattern,
    window_start: date,
    window_end: date,
) -> List[date]:
    """
    Generates projected occurrence dates for a RecurrencePattern strictly within
    (window_start, window_end].
    """
    candidate_dates: List[date] = []

    # 1. Monthly cadence (interval between 28 and 31 days, or preferred day_of_month)
    if 28 <= pat.interval_days <= 31 and pat.day_of_month is not None:
        target_dom = pat.day_of_month
        # Start searching from the month containing window_start
        base_date = date(window_start.year, window_start.month, 1)

        # 90 days spans at most 4-5 calendar months
        for month_offset in range(0, 6):
            dt = add_months_clamped(base_date, month_offset, target_dom)
            if window_start <= dt <= window_end:
                candidate_dates.append(dt)
    else:
        # 2. General interval cadence (e.g. 5, 7, 10, 14, 21 days)
        interval = max(1, pat.interval_days)
        cur_date = pat.last_event_date + timedelta(days=interval)

        # Fast-forward to window_start if necessary
        if cur_date < window_start:
            days_behind = (window_start - cur_date).days
            steps = (days_behind // interval) + 1
            cur_date += timedelta(days=steps * interval)

        while cur_date <= window_end:
            if cur_date >= window_start:
                candidate_dates.append(cur_date)
            cur_date += timedelta(days=interval)

    candidate_dates.sort()
    return candidate_dates


def forecast_cash_flows(
    request: Dict[str, Any],
    user_events: List[CanonicalEvent],
    user_profile: Dict[str, Any],
    stage2_facts: Optional[List[Any]] = None,
    forecast_days: int = config.FORECAST_DAYS,
) -> List[CanonicalEvent]:
    """
    Produces the per-request 90-day forecasted cash-flow timeline.

    Parameters:
        request: dict from requests.csv or sample_requests.csv with 'request_date'
        user_events: cleaned + evidence-augmented CanonicalEvents for this user (Stage 1/2)
        user_profile: dict from financial_profiles.csv for this user
        stage2_facts: optional parsed message facts from Stage 2 for this user
        forecast_days: number of days to forecast forward (default 90)

    Returns:
        Chronologically sorted list of CanonicalEvent objects in (request_date, request_date + 90].
    """
    request_date: date = request["request_date"]
    window_start: date = request_date
    window_end: date = request_date + timedelta(days=forecast_days)
    user_id: str = request["user_id"]
    home_currency: str = user_profile.get("home_currency", "USD")

    # ── 1. Defensive Assertion: Profile Category Invariant ───────────────────
    # Zero users have overlap between protected and flexible categories
    def _parse_cats(val: Any) -> Set[str]:
        if not val:
            return set()
        if isinstance(val, (set, list, tuple)):
            return set(val)
        if isinstance(val, str):
            return set(c.strip() for c in val.split("|") if c.strip())
        return set()

    protect_cats = _parse_cats(user_profile.get("expense_categories_to_protect"))
    reduce_cats = _parse_cats(user_profile.get("expense_categories_user_is_willing_to_reduce"))
    stop_cats = _parse_cats(user_profile.get("expense_categories_user_is_willing_to_stop"))
    profile_overlap = protect_cats & (reduce_cats | stop_cats)
    assert not profile_overlap, (
        f"Defensive assertion failed: user {user_id} has profile conflict overlap: {profile_overlap}"
    )

    # ── 2. Filter Explicit Facts in the Forecast Window ──────────────────────
    # Explicit events: scheduled events, pending debits, or message-generated events
    explicit_events: List[CanonicalEvent] = []
    for ev in user_events:
        if not ev.included_in_forecast:
            continue
        ev_date = ev.settlement_date or ev.event_date
        if ev_date is None:
            continue
        if window_start <= ev_date <= window_end:
            # We keep scheduled events, pending debits, and message-generated events
            if ev.status == "scheduled" or (ev.status == "pending" and ev.direction == "debit") or ev.source == "message":
                ev.projected = False
                explicit_events.append(ev)

    # ── 3. Detect Recurrence Patterns ────────────────────────────────────────
    patterns = detect_user_recurrence_patterns(user_events, stage2_facts=stage2_facts)

    # Extract user-specific Stage 2 facts
    user_facts = [f for f in (stage2_facts or []) if getattr(f, "user_id", "") == user_id]

    # Check for one-off adjustments on the next payroll cycle
    unpaid_leave_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "salary_unpaid_leave_reduction"]
    arrears_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "salary_with_arrears"]

    # Determine next confirmed income date for near-term debit amount rule (Fix 2)
    next_income_date: Optional[date] = None
    all_income_dates: List[date] = [
        (ex.settlement_date or ex.event_date)
        for ex in explicit_events
        if ex.direction == "credit" and (ex.settlement_date or ex.event_date) is not None
    ]
    for pat in patterns.values():
        if pat.category == "salary" and not pat.is_suppressed:
            sal_cand = generate_candidate_dates(pat, window_start, window_end)
            if sal_cand:
                all_income_dates.append(sal_cand[0])
    if all_income_dates:
        next_income_date = min(all_income_dates)

    # ── 4. Project Recurrence Patterns with Deduplication ────────────────────
    projected_events: List[CanonicalEvent] = []

    for pattern_key, pat in patterns.items():
        if pat.is_suppressed:
            continue

        cand_dates = generate_candidate_dates(pat, window_start, window_end)
        is_first_occurrence = True

        for proj_date in cand_dates:
            # Check deduplication against explicit facts of same category and direction
            matched_explicit = None
            for ex in explicit_events:
                if ex.category == pat.category and ex.direction == pat.direction:
                    if pat.series_id and (ex.description or "").strip().casefold() not in {
                        pat.series_id, "next confirmed salary"
                    }:
                        continue
                    ex_date = ex.settlement_date or ex.event_date
                    if ex_date is not None and abs((ex_date - proj_date).days) <= DEDUPE_WINDOW_DAYS:
                        matched_explicit = ex
                        break

            if matched_explicit is not None:
                # Explicit fact overrides projection for this occurrence
                # Never double-count!
                is_first_occurrence = False
                continue

            # Amount determination for this projection
            proj_amount = pat.amount

            # Check if this is the first projected salary cycle and has one-off message adjustments
            if pat.category == "salary" and is_first_occurrence:
                if unpaid_leave_facts and unpaid_leave_facts[0].extracted_amounts:
                    # Next salary reduced due to unpaid leave
                    proj_amount = unpaid_leave_facts[0].extracted_amounts[0][1]
                elif arrears_facts and arrears_facts[0].extracted_amounts:
                    # Regular salary + arrears adjustment
                    if len(arrears_facts[0].extracted_amounts) >= 2:
                        reg_amt = arrears_facts[0].extracted_amounts[0][1]
                        arr_amt = arrears_facts[0].extracted_amounts[1][1]
                        proj_amount = reg_amt + arr_amt

            is_first_occurrence = False
            proj_amount = round_money(proj_amount)

            # Create new projected CanonicalEvent
            proj_event = CanonicalEvent(
                event_id=f"proj_{user_id}_{pattern_key.replace(':', '_')}_{proj_date.strftime('%Y%m%d')}",
                user_id=user_id,
                event_type=pat.event_type,
                description=f"Projected {pat.category} ({pat.recurrence_confidence} confidence)",
                category=pat.category,
                direction=pat.direction,
                amount=proj_amount,
                currency=home_currency,
                event_date=proj_date,
                settlement_date=proj_date,
                status="scheduled",
                linked_event_id="",
                flexibility=pat.flexibility,
                minimum_allowed_amount=pat.minimum_allowed_amount,
                source="projected",
                locked=False,
                included_in_forecast=True,
                exclusion_reason=None,
                projected=True,
            )
            projected_events.append(proj_event)

    # ── 5. Combine and Sort Chronologically ──────────────────────────────────
    all_forecast_events = explicit_events + projected_events

    # Sort by date, with credits before debits on the same day for financial conservatism
    all_forecast_events.sort(
        key=lambda e: (
            e.settlement_date or e.event_date or date.min,
            0 if e.direction == "credit" else 1,
            e.event_id,
        )
    )

    return all_forecast_events

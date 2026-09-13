"""
event_cleaner.py -- Stage 1: Event Cleaning

Transforms raw financial_events.csv rows (loaded via data_io, converted to
CanonicalEvent) into a cleaned, deduplicated list per user.  Every row is
kept in the output with ``included_in_forecast`` and ``exclusion_reason``
set so nothing is silently discarded.

This module does NOT:
  - replay history to recompute starting balances (current_available_balance
    from financial_profiles.csv is authoritative as of request_date)
  - parse messages or images (Stage 2)
  - forecast future cash flow (Stage 3+)
  - apply the policy / planner (Stage 4+)

Design principles:
  - One named filter function per status rule (testable independently).
  - One named handler per linked-event pattern (5 patterns, 58 rows total).
  - A reusable conflict-resolution utility for later stages.
  - Explicit validation of flexibility / minimum_allowed_amount invariants.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from canonical import CanonicalEvent
from config import EVENT_STATUSES
from money import convert_to_home_currency, to_decimal

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 0.  RAW ROW  →  CanonicalEvent  CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def raw_to_canonical(
    row: Dict[str, Any],
    home_currency: str,
) -> CanonicalEvent:
    """
    Convert a single raw event dict (from ``data_io.load_financial_events``)
    into a ``CanonicalEvent`` with the amount converted to *home_currency*.

    Blank amounts are preserved as ``None`` (they will be filled from images
    in a later stage).
    """
    raw_amount: Optional[Decimal] = row["amount"]
    raw_min: Optional[Decimal] = row["minimum_allowed_amount"]
    currency: str = row["currency"]
    settlement: Optional[date] = row["settlement_date"]

    # Convert to home currency if amount is present and currencies differ
    hc_amount: Optional[Decimal] = None
    if raw_amount is not None:
        if currency == home_currency:
            hc_amount = raw_amount
        else:
            if settlement is None:
                # Fallback to event_date if settlement_date is blank
                settlement = row["event_date"]
            if settlement is not None:
                hc_amount = convert_to_home_currency(
                    raw_amount, currency, home_currency, settlement,
                )
            else:
                logger.warning(
                    "Event %s: no date available for FX conversion; "
                    "keeping original currency amount.",
                    row["event_id"],
                )
                hc_amount = raw_amount

    hc_min: Optional[Decimal] = None
    if raw_min is not None:
        if currency == home_currency:
            hc_min = raw_min
        else:
            if settlement is not None:
                hc_min = convert_to_home_currency(
                    raw_min, currency, home_currency, settlement,
                )
            else:
                hc_min = raw_min

    return CanonicalEvent(
        event_id=row["event_id"],
        user_id=row["user_id"],
        event_type=row["event_type"],
        description=row["description"],
        category=row["category"],
        direction=row["direction"],
        amount=hc_amount,
        currency=currency,
        event_date=row["event_date"],
        settlement_date=row["settlement_date"],
        status=row["status"],
        linked_event_id=row["linked_event_id"],
        flexibility=row["flexibility"],
        minimum_allowed_amount=hc_min,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1.  STATUS FILTERS  (one named function per rule)
# ═══════════════════════════════════════════════════════════════════════════
#
# Per problem_statement.md §90-Day Safety Check:
#   "Ignore pending credits, failed or cancelled transactions, duplicate
#    records, and unrealized investments."
#
# CRITICAL: "pending credits" means direction=credit AND status=pending.
#           A pending *debit* (upcoming bill) is a real future obligation
#           and must NOT be excluded.

def filter_cancelled(ev: CanonicalEvent) -> Optional[str]:
    """Exclude cancelled transactions."""
    if ev.status == "cancelled":
        return "cancelled_transaction"
    return None


def filter_failed(ev: CanonicalEvent) -> Optional[str]:
    """Exclude failed transactions."""
    if ev.status == "failed":
        return "failed_transaction"
    return None


def filter_pending_credit(ev: CanonicalEvent) -> Optional[str]:
    """
    Exclude pending *credits* only.

    Pending debits (upcoming bills) are real obligations and stay included.
    """
    if ev.status == "pending" and ev.direction == "credit":
        return "pending_credit"
    return None


def filter_unrealized_investment(ev: CanonicalEvent) -> Optional[str]:
    """Exclude unrealized investment valuations (non-cash, not real flow)."""
    if ev.status == "unrealized":
        return "unrealized_investment"
    return None


# Filters that KEEP events (no exclusion reason):
#   - settled:   always kept (real completed cash movement)
#   - scheduled: kept (real future obligation)
#   - pending debit: kept (upcoming bill / payment)

_STATUS_FILTERS = [
    filter_cancelled,
    filter_failed,
    filter_pending_credit,
    filter_unrealized_investment,
]


def apply_status_filters(ev: CanonicalEvent) -> Optional[str]:
    """
    Run all status-based exclusion filters.

    Returns the exclusion reason string if the event should be excluded,
    or ``None`` if it passes all filters (i.e. should be included).
    """
    for filt in _STATUS_FILTERS:
        reason = filt(ev)
        if reason is not None:
            return reason
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LINKED-EVENT RESOLUTION  (one named handler per pattern)
# ═══════════════════════════════════════════════════════════════════════════
#
# Dataset has exactly 58 linked rows in 5 patterns.  The "child" has
# linked_event_id pointing to the "parent".
#
# We return a set of event_ids that should be EXCLUDED by the linked-event
# resolution step.  Reasons are tagged so they're auditable.

LinkagePattern = str  # e.g. "settled->settled"


def _pattern_key(child: CanonicalEvent, parent: CanonicalEvent) -> str:
    return f"{child.status}->{parent.status}"


def resolve_settled_settled(
    child: CanonicalEvent, parent: CanonicalEvent,
) -> Dict[str, Optional[str]]:
    """
    settled -> settled  (19 rows)

    Both are real cash movements (e.g. settled refund linked to settled
    original purchase).  Keep both.
    """
    return {}  # no exclusions


def resolve_pending_settled(
    child: CanonicalEvent, parent: CanonicalEvent,
) -> Dict[str, Optional[str]]:
    """
    pending -> settled  (14 rows)

    Shopping-category events with mixed directions linked to settled originals:
      - 8 rows: pending credit refunds ('Pending merchant refund') linked to
        settled original purchases ('Purchase awaiting refund'). Excluded by
        ``filter_pending_credit`` (unconfirmed credits do not count as cash).
      - 6 rows: pending debit charges ('Possible duplicate card charge') linked to
        settled original charges ('Original card charge'). Kept as conservative
        obligations (pending debits must be reserved).

    No additional exclusion from linkage itself; status filtering properly
    excludes the 8 pending credit refunds while preserving the 6 pending debits.
    """
    return {}  # status filter handles pending credits; pending debits stay


def resolve_unrealized_settled(
    child: CanonicalEvent, parent: CanonicalEvent,
) -> Dict[str, Optional[str]]:
    """
    unrealized -> settled  (10 rows)

    Investment valuation linked to its settled purchase.  Drop the
    unrealized valuation (already handled by status filter), keep the
    settled purchase.
    """
    # The status filter already excludes unrealized, but we confirm here.
    return {child.event_id: "unrealized_valuation_linked_to_settled_purchase"}


def resolve_settled_cancelled(
    child: CanonicalEvent, parent: CanonicalEvent,
) -> Dict[str, Optional[str]]:
    """
    settled -> cancelled  (8 rows)

    A settled retry/replacement linked to an earlier cancelled attempt.
    Keep the settled one; drop the cancelled one.
    The cancelled parent is already excluded by ``filter_cancelled``,
    but we tag it here to confirm the link doesn't cause double-processing.
    """
    return {parent.event_id: "cancelled_attempt_replaced_by_settled"}


def resolve_scheduled_failed(
    child: CanonicalEvent, parent: CanonicalEvent,
) -> Dict[str, Optional[str]]:
    """
    scheduled -> failed  (7 rows)

    A scheduled retry linked to a previous failed payment.
    Keep the scheduled retry as a real future obligation; drop the failed one.
    The failed parent is already excluded by ``filter_failed``.
    """
    return {parent.event_id: "failed_payment_retried_as_scheduled"}


_LINKAGE_HANDLERS = {
    "settled->settled":   resolve_settled_settled,
    "pending->settled":   resolve_pending_settled,
    "unrealized->settled": resolve_unrealized_settled,
    "settled->cancelled": resolve_settled_cancelled,
    "scheduled->failed":  resolve_scheduled_failed,
}


def resolve_all_linked_events(
    events_by_id: Dict[str, CanonicalEvent],
) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    """
    Process all linked events.

    Returns:
        exclusions: dict of {event_id: reason} for events to exclude via linkage.
        unrecognised: list of (child_id, parent_id, pattern) for patterns
                      not in the 5 known handlers.
    """
    exclusions: Dict[str, str] = {}
    unrecognised: List[Tuple[str, str, str]] = []

    for ev in events_by_id.values():
        if not ev.linked_event_id:
            continue
        parent = events_by_id.get(ev.linked_event_id)
        if parent is None:
            logger.warning(
                "Orphan link: %s -> %s (parent not found)",
                ev.event_id, ev.linked_event_id,
            )
            continue

        pattern = _pattern_key(ev, parent)
        handler = _LINKAGE_HANDLERS.get(pattern)

        if handler is None:
            unrecognised.append((ev.event_id, parent.event_id, pattern))
            logger.warning(
                "Unrecognised linkage pattern: %s (%s -> %s)",
                pattern, ev.event_id, parent.event_id,
            )
        else:
            result = handler(ev, parent)
            exclusions.update(result)

    return exclusions, unrecognised


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CONFLICT-RESOLUTION UTILITY
# ═══════════════════════════════════════════════════════════════════════════
#
# Per problem_statement.md, when records conflict, apply:
#   1. Explicit cancellation, settlement, or amendment.
#   2. Newer record from the same source.
#   3. Settled event over an estimate/forecast.
#   4. Financially safer interpretation (last resort).
#
# Exposed as a utility so later stages (message/image overrides) can call it.

_STATUS_AUTHORITY = {
    "settled": 4,       # highest authority — confirmed fact
    "cancelled": 3,     # explicit action
    "failed": 2,
    "scheduled": 1,
    "pending": 1,
    "unrealized": 0,    # lowest — estimate
}


def resolve_conflict(
    event_a: CanonicalEvent,
    event_b: CanonicalEvent,
) -> CanonicalEvent:
    """
    Given two conflicting events about the same financial fact, return the
    one that should be kept per the problem statement's precedence rules.

    Rule 1: explicit cancellation/settlement/amendment wins.
    Rule 2: newer record from the same source wins.
    Rule 3: settled beats estimate/forecast.
    Rule 4: financially safer interpretation (higher amount for debits,
            lower amount for credits).
    """
    # Rule 1 + 3: status authority
    auth_a = _STATUS_AUTHORITY.get(event_a.status, 0)
    auth_b = _STATUS_AUTHORITY.get(event_b.status, 0)
    if auth_a != auth_b:
        return event_a if auth_a > auth_b else event_b

    # Rule 2: newer record (by settlement_date, then event_date)
    date_a = event_a.settlement_date or event_a.event_date
    date_b = event_b.settlement_date or event_b.event_date
    if date_a and date_b and date_a != date_b:
        return event_a if date_a > date_b else event_b

    # Rule 4: financially safer interpretation
    if event_a.amount is not None and event_b.amount is not None:
        if event_a.direction == "debit":
            # Higher debit = more conservative
            return event_a if event_a.amount >= event_b.amount else event_b
        else:
            # Lower credit = more conservative
            return event_a if event_a.amount <= event_b.amount else event_b

    # Tie-break: keep first (stable)
    return event_a


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FLEXIBILITY / MINIMUM_ALLOWED_AMOUNT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Dataset invariants (verified in Stage 0):
#   - fixed, stoppable:               minimum_allowed_amount is always None
#   - reducible, reducible_or_stoppable: minimum_allowed_amount is always set
#
# Breaking this invariant is a hard error — downstream spending-change
# candidate generation depends on it.

def validate_flexibility(ev: CanonicalEvent) -> None:
    """
    Raise ``ValueError`` if a reducible event is missing its
    ``minimum_allowed_amount`` floor, or a fixed/stoppable event
    unexpectedly has one.
    """
    if ev.flexibility in ("reducible", "reducible_or_stoppable"):
        if ev.minimum_allowed_amount is None:
            raise ValueError(
                f"Event {ev.event_id}: flexibility={ev.flexibility!r} "
                f"but minimum_allowed_amount is None. This would silently "
                f"break spending-change candidate generation."
            )
    elif ev.flexibility in ("fixed", "stoppable"):
        if ev.minimum_allowed_amount is not None:
            raise ValueError(
                f"Event {ev.event_id}: flexibility={ev.flexibility!r} "
                f"but minimum_allowed_amount={ev.minimum_allowed_amount}. "
                f"Unexpected — check data."
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BALANCE AUTHORITY BOUNDARY
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT:  financial_profiles.csv's ``current_available_balance`` is
# authoritative as of ``request_date``.  Event cleaning NEVER replays
# historical settled events to reconstruct a starting balance.
#
# The cleaned event list is used ONLY for:
#   (a) Forward-looking events after request_date (forecast).
#   (b) Recurrence-pattern inference from history.
#
# This boundary must not be violated by any later stage.

BALANCE_AUTHORITY_NOTE = (
    "current_available_balance from financial_profiles.csv is the "
    "authoritative starting balance as of request_date.  Do NOT "
    "replay settled history to recompute it."
)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN CLEANING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def clean_events(
    raw_events: List[Dict[str, Any]],
    profiles: List[Dict[str, Any]],
) -> List[CanonicalEvent]:
    """
    Full Stage 1 cleaning pipeline.

    1. Convert raw rows to CanonicalEvent (home-currency amounts).
    2. Validate flexibility invariants.
    3. Apply status filters.
    4. Resolve linked-event patterns.
    5. Tag each event with ``included_in_forecast`` and ``exclusion_reason``.

    Returns ALL events (included and excluded) — nothing is silently dropped.
    The caller filters on ``included_in_forecast`` when needed.
    """
    # Build user -> home_currency lookup
    user_home: Dict[str, str] = {}
    if profiles:
        if isinstance(profiles, dict):
            for uid, p in profiles.items():
                user_home[uid] = p.get("home_currency") if isinstance(p, dict) else getattr(p, "home_currency", "")
        else:
            for p in profiles:
                user_home[p["user_id"]] = p["home_currency"]

    # Step 1: Convert to CanonicalEvent if not already canonical
    canonical_events: List[CanonicalEvent] = []
    for row in raw_events:
        if isinstance(row, CanonicalEvent):
            canonical_events.append(row)
        else:
            hc = user_home.get(row["user_id"], row["currency"])
            canonical_events.append(raw_to_canonical(row, hc))

    # Step 2: Validate flexibility invariants
    for ev in canonical_events:
        validate_flexibility(ev)

    # Step 3: Apply status filters
    for ev in canonical_events:
        reason = apply_status_filters(ev)
        if reason is not None:
            ev.included_in_forecast = False
            ev.exclusion_reason = reason

    # Step 4: Resolve linked events
    events_by_id = {ev.event_id: ev for ev in canonical_events}
    link_exclusions, unrecognised = resolve_all_linked_events(events_by_id)

    for eid, reason in link_exclusions.items():
        ev = events_by_id.get(eid)
        if ev is not None and ev.included_in_forecast:
            # Only override if not already excluded by status filter
            ev.included_in_forecast = False
            ev.exclusion_reason = reason

    if unrecognised:
        logger.error(
            "Found %d unrecognised linkage patterns: %s",
            len(unrecognised), unrecognised,
        )

    # Log summary
    included = sum(1 for ev in canonical_events if ev.included_in_forecast)
    excluded = len(canonical_events) - included
    logger.info(
        "Event cleaning complete: %d total, %d included, %d excluded. "
        "%s",
        len(canonical_events), included, excluded,
        BALANCE_AUTHORITY_NOTE,
    )

    return canonical_events


def get_user_events(
    all_events: List[CanonicalEvent],
    user_id: str,
    *,
    forecast_only: bool = True,
) -> List[CanonicalEvent]:
    """Convenience: filter events for a single user."""
    return [
        ev for ev in all_events
        if ev.user_id == user_id
        and (not forecast_only or ev.included_in_forecast)
    ]

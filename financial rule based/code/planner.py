"""Stage 5 candidate generation.

Produces fully specified, replay-verified choices from the Stage 4 timeline.
This module deliberately does not rank candidates or produce final output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config
from canonical import CanonicalEvent
from deadline_filter import filter_by_deadline, filter_by_installment_cap, is_method_accepted
from money import round_money
from simulator import simulate_timeline


Payment = Tuple[date, Decimal]


@dataclass(frozen=True)
class Candidate:
    """A safe, complete plan ready for Stage 6 ranking."""
    candidate_type: str
    method: str
    payments: Tuple[Payment, ...]
    payment_option_id: Optional[str]
    total_paid: Decimal
    spending_adjustments: Dict[str, Optional[Decimal]]
    spending_changes: Tuple[str, ...]
    is_fallback: bool = False


@dataclass
class CandidateGenerationResult:
    candidates: List[Candidate] = field(default_factory=list)
    filtered: List[Tuple[str, str]] = field(default_factory=list)


def _replay_is_safe(
    request: Dict[str, Any], profile: Dict[str, Any], timeline: List[CanonicalEvent],
    payments: Iterable[Payment], adjustments: Optional[Dict[str, Optional[Decimal]]] = None,
) -> bool:
    payment_list = list(payments)
    last_date = max((payment_date for payment_date, _ in payment_list), default=request["request_date"])
    forecast_days = max(config.FORECAST_DAYS, (last_date - request["request_date"]).days)
    return simulate_timeline(
        starting_balance=profile["current_available_balance"],
        minimum_balance_to_keep=profile["minimum_balance_to_keep"],
        forecast_events=timeline,
        request_date=request["request_date"],
        forecast_days=forecast_days,
        proposed_payments=payment_list,
        spending_adjustments=adjustments,
    ).is_safe


def _option_payments(option: Dict[str, Any]) -> Tuple[Payment, ...]:
    count = option["number_of_payments"]
    first = option["first_payment_date"]
    assert count is not None and first is not None, f"Malformed option {option['payment_option_id']}"
    if count == 1:
        return ((first, option["payment_amount"]),)
    frequency = option["payment_frequency_days"]
    assert frequency, f"Missing frequency for option {option['payment_option_id']}"
    from datetime import timedelta
    return tuple((first + timedelta(days=index * int(frequency)), option["payment_amount"]) for index in range(count))


def _validate_full_payment_option(option: Dict[str, Any]) -> None:
    """Make the documented dataset invariant executable rather than implicit."""
    if option["payment_method"] == "full_payment":
        assert option["number_of_payments"] == 1, f"Full payment option has multiple payments: {option['payment_option_id']}"
        assert option["financing_fee"] == Decimal("0"), f"Full payment option has a fee: {option['payment_option_id']}"


def _parse_categories(value: Any) -> set[str]:
    return {part.strip() for part in (value or "").split("|") if part.strip()}


@dataclass(frozen=True)
class _AdjustmentAction:
    adjustment_key: str
    event_id: str
    new_amount: Optional[Decimal]
    display: str
    saving: Decimal


def _eligible_adjustments(
    profile: Dict[str, Any], timeline: List[CanonicalEvent], source_events: List[CanonicalEvent], request_date: date,
) -> List[_AdjustmentAction]:
    protected = _parse_categories(profile.get("expense_categories_to_protect"))
    reducible = _parse_categories(profile.get("expense_categories_user_is_willing_to_reduce"))
    stoppable = _parse_categories(profile.get("expense_categories_user_is_willing_to_stop"))
    assert not (protected & (reducible | stoppable)), "Protected and adjustable categories overlap"

    # Output changes must cite a supplied financial-event id, never a synthetic
    # ``proj_*`` timeline id.  The category key still applies the adjustment to
    # every projected occurrence of that recurring expense in the simulator.
    representatives: Dict[str, List[CanonicalEvent]] = {}
    for event in source_events:
        event_date = event.settlement_date or event.event_date
        if (
            event.direction == "debit" and event.included_in_forecast and event_date is not None
            and event_date <= request_date and not event.projected
        ):
            representatives.setdefault(event.category, []).append(event)
    for category in representatives:
        representatives[category].sort(key=lambda event: event.settlement_date or event.event_date or date.min, reverse=True)

    def representative_for(category: str, allowed_flexibility: set[str], needs_minimum: bool) -> Optional[CanonicalEvent]:
        for event in representatives.get(category, []):
            if event.flexibility in allowed_flexibility and (not needs_minimum or event.minimum_allowed_amount is not None):
                return event
        return None

    actions: List[_AdjustmentAction] = []
    seen_categories: set[str] = set()
    for event in timeline:
        if event.direction != "debit" or event.amount is None or event.category in protected:
            continue
        if event.category in seen_categories or event.category not in representatives:
            continue
        seen_categories.add(event.category)
        if event.category in stoppable and event.flexibility in {"stoppable", "reducible_or_stoppable"}:
            representative = representative_for(event.category, {"stoppable", "reducible_or_stoppable"}, False)
            if representative is not None:
                actions.append(_AdjustmentAction(
                    event.category, representative.event_id, None, f"stop:{representative.event_id}", event.amount,
                ))
        if (
            event.category in reducible
            and event.flexibility in {"reducible", "reducible_or_stoppable"}
            and event.minimum_allowed_amount is not None
            and event.minimum_allowed_amount < event.amount
        ):
            representative = representative_for(event.category, {"reducible", "reducible_or_stoppable"}, True)
            if representative is not None:
                new_amount = event.minimum_allowed_amount
                actions.append(_AdjustmentAction(
                    event.category, representative.event_id, new_amount,
                    f"reduce_to:{representative.event_id}:{new_amount}", event.amount - new_amount,
                ))
    return sorted(actions, key=lambda action: (action.saving, action.event_id, action.display), reverse=True)


def _best_spending_change_candidate(
    request: Dict[str, Any], profile: Dict[str, Any], timeline: List[CanonicalEvent], source_events: List[CanonicalEvent],
) -> Optional[Candidate]:
    """Find the least-saving bundle of one to three permitted event changes that is safe."""
    actions = _eligible_adjustments(profile, timeline, source_events, request["request_date"])
    payment = ((request["request_date"], request["requested_amount"]),)
    feasible: List[Tuple[Decimal, Tuple[_AdjustmentAction, ...]]] = []
    for size in range(1, min(3, len(actions)) + 1):
        for selected in combinations(actions, size):
            # The same event cannot be stopped and reduced in one candidate.
            if len({action.adjustment_key for action in selected}) != len(selected):
                continue
            adjustments = {action.adjustment_key: action.new_amount for action in selected}
            if _replay_is_safe(request, profile, timeline, payment, adjustments):
                feasible.append((sum((action.saving for action in selected), Decimal("0")), selected))
        if feasible:
            saving, selected = min(feasible, key=lambda pair: (pair[0], tuple(a.display for a in pair[1])))
            adjustments = {action.adjustment_key: action.new_amount for action in selected}
            return Candidate(
                candidate_type="full_payment_with_spending_changes",
                method="full_payment",
                payments=payment,
                payment_option_id=None,
                total_paid=request["requested_amount"],
                spending_adjustments=adjustments,
                spending_changes=tuple(action.display for action in selected),
            )
    return None


def generate_candidates(
    request: Dict[str, Any], profile: Dict[str, Any], timeline: List[CanonicalEvent],
    payment_options: List[Dict[str, Any]], amount_safe_to_pay: Decimal,
    earliest_date_for_full_payment: Optional[date], source_events: List[CanonicalEvent],
) -> CandidateGenerationResult:
    """Generate every basic Stage 5 candidate that survives eligibility and replay safety."""
    result = CandidateGenerationResult()
    requested = request["requested_amount"]
    today_payment = ((request["request_date"], requested),)

    if is_method_accepted("full_payment", request, profile):
        if amount_safe_to_pay >= requested and _replay_is_safe(request, profile, timeline, today_payment):
            result.candidates.append(Candidate("full_payment_today", "full_payment", today_payment, None, requested, {}, ()))
        elif amount_safe_to_pay < requested:
            result.filtered.append(("full_payment", "unsafe-on-replay"))
    else:
        result.filtered.append(("full_payment", "method-not-accepted"))

    for option in payment_options:
        _validate_full_payment_option(option)
        if option["payment_method"] != "installments":
            continue
        option_id = option["payment_option_id"]
        if not is_method_accepted("installments", request, profile):
            result.filtered.append((option_id, "method-not-accepted"))
            continue
        if not filter_by_deadline(option, request):
            result.filtered.append((option_id, "deadline"))
            continue
        if not filter_by_installment_cap(option, profile):
            result.filtered.append((option_id, "installment-cap"))
            continue
        payments = _option_payments(option)
        if not _replay_is_safe(request, profile, timeline, payments):
            result.filtered.append((option_id, "unsafe-on-replay"))
            continue
        assert sum((amount for _, amount in payments), Decimal("0")) == option["total_payable_amount"], option_id
        result.candidates.append(Candidate("installments", "installments", payments, option_id, option["total_payable_amount"], {}, ()))

    if not is_method_accepted("partial_payment", request, profile):
        result.filtered.append(("partial_payment", "partial-flag-false" if not request["allows_partial_payment"] else "method-not-accepted"))
    elif not (Decimal("0") < amount_safe_to_pay < requested):
        result.filtered.append(("partial_payment", "not-a-partial-amount"))
    elif earliest_date_for_full_payment is None or earliest_date_for_full_payment > request["desired_completion_date"]:
        result.filtered.append(("partial_payment", "completion-after-deadline"))
    else:
        remainder = round_money(requested - amount_safe_to_pay)
        payments = ((request["request_date"], amount_safe_to_pay), (earliest_date_for_full_payment, remainder))
        assert sum((amount for _, amount in payments), Decimal("0")) == requested
        if _replay_is_safe(request, profile, timeline, payments):
            result.candidates.append(Candidate("partial_payment", "partial_payment", payments, None, requested, {}, ()))
        else:
            result.filtered.append(("partial_payment", "unsafe-on-replay"))

    if not is_method_accepted("full_payment", request, profile):
        result.filtered.append(("wait", "method-not-accepted"))
    elif earliest_date_for_full_payment == request["request_date"]:
        result.filtered.append(("wait", "safe-today-not-wait"))
    elif earliest_date_for_full_payment is None or earliest_date_for_full_payment > request["desired_completion_date"]:
        result.filtered.append(("wait", "completion-after-deadline"))
    else:
        payments = ((earliest_date_for_full_payment, requested),)
        if _replay_is_safe(request, profile, timeline, payments):
            result.candidates.append(Candidate("wait", "wait", payments, None, requested, {}, ()))
        else:
            result.filtered.append(("wait", "unsafe-on-replay"))

    if is_method_accepted("full_payment", request, profile) and amount_safe_to_pay < requested:
        adjusted = _best_spending_change_candidate(request, profile, timeline, source_events)
        if adjusted is not None:
            result.candidates.append(adjusted)
        else:
            result.filtered.append(("full_payment_with_spending_changes", "unsafe-on-replay"))

    # A guaranteed fallback is intentionally not replayed: it makes no payment.
    result.candidates.append(Candidate("not_recommended", "not_recommended", (), None, Decimal("0"), {}, (), True))
    return result

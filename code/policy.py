"""Stage 6 deterministic decision policy.

Ranks only replay-safe candidates produced by Stage 5.  It does not compose
explanations or write output files; those remain downstream responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Optional, Tuple

from planner import Candidate


@dataclass(frozen=True)
class PolicyDecision:
    """The selected safe candidate and its specification-defined status."""
    candidate: Candidate
    affordability_status: str
    ranking_key: Tuple[object, ...]


def candidate_completes_by_deadline(candidate: Candidate, deadline: date) -> bool:
    """A fallback never completes; all payment dates must be by the deadline."""
    return not candidate.is_fallback and bool(candidate.payments) and max(day for day, _ in candidate.payments) <= deadline


def payment_option_sort_key(payment_option_id: Optional[str]) -> Tuple[int, str]:
    """Order `payment_option_<number>` by its numeric suffix, never by text."""
    if payment_option_id is None:
        return (-1, "")
    match = re.fullmatch(r"payment_option_(\d+)", payment_option_id)
    assert match is not None, f"Unexpected payment option id: {payment_option_id}"
    return (int(match.group(1)), payment_option_id)


def candidate_ranking_key(candidate: Candidate, deadline: date) -> Tuple[object, ...]:
    """The required lexicographic ranking, followed by deterministic stability."""
    if candidate.is_fallback:
        return (1, 1, float("inf"), date.max, float("inf"), "~", candidate.candidate_type)
    assert candidate.payments, f"Non-fallback candidate has no payments: {candidate}"
    return (
        0 if candidate_completes_by_deadline(candidate, deadline) else 1,
        0 if not candidate.spending_changes else 1,
        candidate.total_paid,
        min(day for day, _ in candidate.payments),
        len(candidate.payments),
        payment_option_sort_key(candidate.payment_option_id),
        candidate.candidate_type,
    )


def affordability_status(candidate: Candidate, request_date: date) -> str:
    """Map a selected Stage 5 candidate to the allowed affordability status."""
    if candidate.is_fallback:
        return "not_affordable"
    if candidate.method == "wait":
        return "affordable_later"
    if candidate.candidate_type == "full_payment_today" and not candidate.spending_changes:
        return "affordable_now"
    return "affordable_with_plan"


def select_best_candidate(candidates: Iterable[Candidate], request_date: date, deadline: date) -> PolicyDecision:
    """Choose one candidate using the six priorities in problem_statement.md."""
    choices = list(candidates)
    assert choices, "Stage 5 must supply at least the not_recommended fallback"
    selected = min(choices, key=lambda candidate: candidate_ranking_key(candidate, deadline))
    return PolicyDecision(selected, affordability_status(selected, request_date), candidate_ranking_key(selected, deadline))

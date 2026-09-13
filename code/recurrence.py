"""
recurrence.py — Stage 3: Recurrence Pattern Detection.

Groups cleaned events by (user_id, category) and infers recurrence patterns:
  - Interval: mode of day-gaps between consecutive occurrences.
  - Recurrence confidence: 'high' if gap std <= 1.0 day, 'low' if std > 1.0 day.
  - Amount:
      * CV = 0 categories (cloud_storage, debt_repayment, delivery_membership,
        education, family_support, insurance, housing, gym, rent,
        music_subscription, work_expense, streaming): latest observed amount.
      * CV = 6-9% categories (healthcare, entertainment, utilities, shopping):
        median of last 3 occurrences.
      * Salary: explicit confirmed/scheduled next event or Stage 2 message override,
        else latest observed settled amount directly (never averaged).
      * CV = 15-16% categories (dining, groceries, transport): median of last 3 occurrences.
      * Investment: NOT recurring; never projected per problem specification.
  - Occurrence threshold: >= 3 occurrences required, unless known fixed-cadence
    category (rent, salary, debt_repayment, insurance, subscriptions, etc.)
    which can be inferred with >= 1 occurrence.
  - Carries flexibility and minimum_allowed_amount forward unchanged.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from canonical import CanonicalEvent
from money import round_money

logger = logging.getLogger(__name__)

# ── Categories with CV = 0.0% (exact fixed amounts) ───────────────────────────
CV_ZERO_CATEGORIES = frozenset({
    "cloud_storage",
    "debt_repayment",
    "delivery_membership",
    "education",
    "family_support",
    "insurance",
    "housing",
    "gym",
    "rent",
    "music_subscription",
    "work_expense",
    "streaming",
})

# ── Categories with CV = 6-9% ────────────────────────────────────────────────
CV_MODERATE_CATEGORIES = frozenset({
    "healthcare",
    "entertainment",
    "utilities",
    "shopping",
    "salary",
})

# ── Categories with CV = 15-16% ──────────────────────────────────────────────
CV_VARIABLE_CATEGORIES = frozenset({
    "dining",
    "groceries",
    "transport",
})

# ── Known fixed-cadence categories (can infer with < 3 occurrences) ──────────
FIXED_CADENCE_CATEGORIES = frozenset({
    "rent",
    "salary",
    "debt_repayment",
    "insurance",
    "housing",
    "family_support",
    "education",
    "streaming",
    "cloud_storage",
    "delivery_membership",
    "gym",
    "music_subscription",
})


@dataclass
class RecurrencePattern:
    """Inferred recurrence pattern for a single (user_id, category) series."""
    user_id: str
    category: str
    direction: str                          # 'credit' | 'debit'
    event_type: str                         # 'expense', 'income', 'subscription', etc.
    interval_days: int                      # mode of day-gaps between consecutive events
    observed_gap_std: float                 # sample standard deviation of gaps (ddof=1)
    recurrence_confidence: str              # 'high' (std <= 1.0) | 'low' (std > 1.0)
    amount: Decimal                         # estimated recurring amount (home currency)
    amount_cv: float                        # amount standard deviation / mean
    flexibility: str                        # 'fixed' | 'reducible' | 'stoppable' | 'reducible_or_stoppable'
    minimum_allowed_amount: Optional[Decimal]  # floor for reducible, None for fixed/stoppable
    last_event_date: date                   # most recent historical event date
    occurrences_count: int                  # number of historical occurrences
    day_of_month: Optional[int] = None      # mode day of month for monthly cadence
    is_suppressed: bool = False             # True if employment ended / suppressed
    override_note: Optional[str] = None     # explanation of any Stage 2 override applied
    latest_amount: Optional[Decimal] = None # latest single observed settled amount
    # Non-empty only when one category contains independently recurring streams.
    # Salary is the important case: two household incomes must not be interleaved.
    series_id: str = ""


def compute_mode_and_std_gaps(dates: List[date]) -> Tuple[int, float, str, Optional[int]]:
    """
    Computes mode of day-gaps, gap sample standard deviation (ddof=1),
    confidence level ('high' vs 'low'), and mode day-of-month.
    """
    if len(dates) < 2:
        dom = dates[0].day if dates else None
        return 30, 0.0, "high", dom

    # Sort dates chronologically
    sorted_dates = sorted(dates)
    gaps = [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]

    # Mode of gaps (most common gap)
    gap_counts = Counter(gaps)
    mode_gap = gap_counts.most_common(1)[0][0]

    # Sample standard deviation (ddof=1)
    if len(gaps) >= 2:
        gap_std = statistics.stdev(gaps)
    else:
        gap_std = 0.0

    confidence = "high" if gap_std <= 1.0 else "low"

    # Preferred day of month
    doms = [d.day for d in sorted_dates]
    dom_mode = Counter(doms).most_common(1)[0][0]

    return mode_gap, gap_std, confidence, dom_mode


def compute_amount_cv(amounts: List[Decimal]) -> float:
    """Computes coefficient of variation (std / mean) for a list of Decimal amounts."""
    if len(amounts) < 2:
        return 0.0
    float_amts = [float(a) for a in amounts]
    mean_val = statistics.mean(float_amts)
    if mean_val == 0.0:
        return 0.0
    std_val = statistics.stdev(float_amts)
    return std_val / mean_val


def detect_series_pattern(
    events: List[CanonicalEvent],
    user_id: str,
    category: str,
    stage2_facts: Optional[List[Any]] = None,
    series_id: str = "",
) -> Optional[RecurrencePattern]:
    """
    Infers a RecurrencePattern for a specific (user_id, category) series.
    Returns None if series is too sparse to project or category is investment.
    """
    # 1. Investment is NEVER recurring per problem statement
    if category == "investment":
        return None

    # 2. Filter to usable events for pattern inference:
    # settled history + validated scheduled debits/credits + pending debits
    expected_direction = "credit" if category == "salary" else "debit"
    usable = [
        e for e in events
        if e.user_id == user_id
        and e.category == category
        and e.direction == expected_direction
        and e.event_type != "refund"
        and e.included_in_forecast
        and e.amount is not None
        and (
            e.status == "settled"
            or e.status == "scheduled"
            or (e.status == "pending" and e.direction == "debit")
        )
    ]

    if not usable:
        return None

    # Sort chronologically
    usable.sort(key=lambda e: (e.settlement_date or e.event_date or date.min))

    # 3. Minimum occurrence threshold check
    is_fixed_cadence = category in FIXED_CADENCE_CATEGORIES
    min_required = 1 if is_fixed_cadence else 3
    if len(usable) < min_required:
        return None

    # 4. Interval and cadence inference
    dates = [
        e.settlement_date or e.event_date
        for e in usable
        if (e.settlement_date or e.event_date) is not None
    ]
    mode_gap, gap_std, confidence, dom = compute_mode_and_std_gaps(dates)

    # 5. Extract series metadata (direction, event_type, flexibility)
    latest_event = usable[-1]
    direction = latest_event.direction
    event_type = latest_event.event_type
    flexibility = latest_event.flexibility
    min_allowed = latest_event.minimum_allowed_amount
    last_date = latest_event.settlement_date or latest_event.event_date or date.today()

    # 6. Amount calculation based on verified category CV
    amounts = [e.amount for e in usable if e.amount is not None]
    cv = compute_amount_cv(amounts)
    override_note = None
    is_suppressed = False

    # Check for Stage 2 overrides matching this user & category
    user_facts = [f for f in (stage2_facts or []) if getattr(f, "user_id", "") == user_id]

    if category == "salary":
        # Check Stage 2 facts for salary changes
        salary_raise_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "salary_raise"]
        unpaid_leave_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "salary_unpaid_leave_reduction"]
        temp_reduction_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "salary_temporary_reduction"]
        emp_ended_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "employment_ended"]
        gig_pending_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "gig_payout_pending"]

        # Check if the most recent historical salary event was a "Final employer payroll"
        last_desc = (latest_event.description or "").lower()
        if "final employer payroll" in last_desc or "final payroll" in last_desc:
            recurring_amount = Decimal("0")
            is_suppressed = True
            override_note = "Final employer payroll settled: employment ended"
        elif emp_ended_facts:
            f = emp_ended_facts[0]
            # Check if remaining household salary is specified
            if f.extracted_amounts:
                # Remaining monthly salary specified
                recurring_amount = f.extracted_amounts[0][1]
                override_note = f"Stage 2 employment ended: remaining household salary {recurring_amount}"
            else:
                # Check target description cluster (Fix 1)
                target_desc = getattr(f, "target_description", None)
                if target_desc == "seasonal":
                    # Only seasonal series ends; check if user has other salary series
                    non_seasonal = [
                        e for e in usable
                        if "seasonal" not in (e.description or "").lower()
                        and "musiman" not in (e.description or "").lower()
                    ]
                    if non_seasonal:
                        # Non-seasonal series continues uninterrupted (e.g. user_12)
                        latest_active = non_seasonal[-1]
                        recurring_amount = latest_active.amount
                        override_note = f"Stage 2 seasonal contract ended; active series continues: {latest_active.description}"
                    else:
                        recurring_amount = Decimal("0")
                        is_suppressed = True
                        override_note = "Stage 2 seasonal contract ended: salary suppressed"
                else:
                    # Employment completely ended: suppress salary
                    recurring_amount = Decimal("0")
                    is_suppressed = True
                    override_note = "Stage 2 employment ended: salary suppressed"
        elif gig_pending_facts:
            # Fix 3: Platform gig payout is pending and unconfirmed
            recurring_amount = Decimal("0")
            is_suppressed = True
            override_note = "Stage 2 gig payout pending: unconfirmed income excluded"
        elif unpaid_leave_facts:
            # The message controls only the next payroll.  Do not let an
            # atypical immediately preceding settled payroll become the
            # permanent post-message baseline: retain the established salary
            # level and let forecaster.py splice the one-cycle amount below.
            settled_salaries = [e.amount for e in usable if e.status == "settled" and e.amount is not None]
            recurring_amount = Counter(settled_salaries).most_common(1)[0][0]
            override_note = "Stage 2 unpaid-leave adjustment: retained established post-cycle salary baseline"
        elif salary_raise_facts and salary_raise_facts[0].extracted_amounts:
            f = salary_raise_facts[0]
            recurring_amount = f.extracted_amounts[0][1]
            override_note = f"Stage 2 salary raise to {recurring_amount}"
        elif temp_reduction_facts and temp_reduction_facts[0].extracted_amounts:
            f = temp_reduction_facts[0]
            recurring_amount = f.extracted_amounts[0][1]
            override_note = f"Stage 2 temporary salary reduction to {recurring_amount}"
        else:
            # Check if there is an explicitly scheduled next salary event in usable
            sched_salary = [e for e in usable if e.status == "scheduled"]
            if sched_salary and sched_salary[-1].amount is not None:
                recurring_amount = sched_salary[-1].amount
                override_note = "Using confirmed scheduled next-salary amount"
            else:
                # A later confirmed payroll supersedes an earlier one.  Salary
                # changes are common; a historical mode would retain an
                # obsolete amount and violates the stated newer-record rule.
                settled_salaries = [e.amount for e in usable if e.status == "settled" and e.amount is not None]
                if settled_salaries:
                    recurring_amount = settled_salaries[-1]
                else:
                    recurring_amount = amounts[-1]
    elif category in CV_ZERO_CATEGORIES:
        # Base amount from most common settled amount (avoids 1-off scheduled settlement outliers)
        settled_amts = [e.amount for e in usable if e.status == "settled" and e.amount is not None]
        if settled_amts:
            recurring_amount = Counter(settled_amts).most_common(1)[0][0]
        else:
            recurring_amount = amounts[-1]

        # Check for Stage 2 lease increase on rent
        if category == "rent":
            lease_facts = [f for f in user_facts if getattr(f, "pattern_family", "") == "lease_rent_increase"]
            if lease_facts:
                # 12% increase on monthly rent
                increased = round_money(recurring_amount * Decimal("1.12"))
                override_note = f"Stage 2 lease increase: {recurring_amount} -> {increased} (+12%)"
                recurring_amount = increased
    elif category in CV_MODERATE_CATEGORIES or category in CV_VARIABLE_CATEGORIES:
        # Median of last 3 occurrences
        last_3 = amounts[-3:]
        median_val = statistics.median(last_3)
        recurring_amount = round_money(Decimal(str(median_val)))
    else:
        # Fallback: median of last 3 or latest
        if len(amounts) >= 3:
            median_val = statistics.median(amounts[-3:])
            recurring_amount = round_money(Decimal(str(median_val)))
        else:
            recurring_amount = amounts[-1]

    return RecurrencePattern(
        user_id=user_id,
        category=category,
        direction=direction,
        event_type=event_type,
        interval_days=mode_gap,
        observed_gap_std=gap_std,
        recurrence_confidence=confidence,
        amount=recurring_amount,
        amount_cv=cv,
        flexibility=flexibility,
        minimum_allowed_amount=min_allowed,
        last_event_date=last_date,
        occurrences_count=len(usable),
        day_of_month=dom,
        is_suppressed=is_suppressed,
        override_note=override_note,
        latest_amount=amounts[-1] if amounts else recurring_amount,
        series_id=series_id,
    )


def detect_user_recurrence_patterns(
    user_events: List[CanonicalEvent],
    stage2_facts: Optional[List[Any]] = None,
) -> Dict[str, RecurrencePattern]:
    """
    Detects all recurring patterns for a single user, keyed by category.
    """
    if not user_events:
        return {}

    user_id = user_events[0].user_id
    categories = sorted(set(e.category for e in user_events))
    patterns: Dict[str, RecurrencePattern] = {}

    for cat in categories:
        # A household can have more than one salary stream.  Inferring one cadence
        # from their interleaved payments turns monthly salaries on (say) the 15th
        # and 20th into a fictitious five-day salary.  Split only when settled
        # history proves distinct description streams; ordinary single-stream
        # salary remains keyed as simply ``salary`` for compatibility.
        if cat == "salary":
            settled = [
                e for e in user_events
                if e.category == "salary" and e.direction == "credit" and e.status == "settled"
            ]
            groups: Dict[str, List[CanonicalEvent]] = defaultdict(list)
            for event in settled:
                key = (event.description or "salary").strip().casefold()
                groups[key].append(event)
            recurring_groups = {key: rows for key, rows in groups.items() if len(rows) >= 2}
            if len(recurring_groups) > 1:
                scheduled = [
                    e for e in user_events
                    if e.category == "salary" and e.direction == "credit" and e.status == "scheduled"
                ]
                for series_id, rows in recurring_groups.items():
                    # A confirmed next payroll normally has a generic description.
                    # Attach it only when its amount identifies exactly one stream.
                    row_amounts = {e.amount for e in rows}
                    series_events = list(rows)
                    for event in scheduled:
                        matches = [
                            key for key, candidate_rows in recurring_groups.items()
                            if event.amount in {candidate.amount for candidate in candidate_rows}
                        ]
                        if matches == [series_id]:
                            series_events.append(event)
                    pat = detect_series_pattern(
                        series_events, user_id, cat, stage2_facts=stage2_facts, series_id=series_id
                    )
                    if pat is not None and not pat.is_suppressed:
                        patterns[f"salary:{series_id}"] = pat
                continue

        pat = detect_series_pattern(user_events, user_id, cat, stage2_facts=stage2_facts)
        if pat is not None and not pat.is_suppressed:
            patterns[cat] = pat

    return patterns


def detect_all_recurrence_patterns(
    all_events: List[CanonicalEvent],
    stage2_facts: Optional[List[Any]] = None,
) -> Dict[Tuple[str, str], RecurrencePattern]:
    """
    Detects recurrence patterns across all (user_id, category) series in the dataset.
    Returns dict keyed by (user_id, category).
    """
    events_by_user_cat: Dict[Tuple[str, str], List[CanonicalEvent]] = defaultdict(list)
    for e in all_events:
        events_by_user_cat[(e.user_id, e.category)].append(e)

    all_patterns: Dict[Tuple[str, str], RecurrencePattern] = {}
    for (uid, cat), evs in events_by_user_cat.items():
        pat = detect_series_pattern(evs, uid, cat, stage2_facts=stage2_facts)
        if pat is not None:
            all_patterns[(uid, cat)] = pat

    return all_patterns

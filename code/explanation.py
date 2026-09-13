"""
explanation.py — Stage 7: Deterministic Decision Explanation Generator.

Produces concise, grounded natural language explanations for financial recommendations
strictly adhering to the templates demonstrated in sample_requests.csv and problem_statement.md.
Zero non-deterministic language models in the decision/explanation path.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from canonical import CanonicalEvent
from policy import PolicyDecision


def format_money(amount: Decimal) -> str:
    """Format decimal amount with thousands separators and optional decimals."""
    if amount == amount.to_integral():
        return f"{int(amount):,}"
    return f"{amount:,.2f}"


def format_date_human(d: date) -> str:
    """Format date as 'D Month YYYY' (e.g. '8 August 2025')."""
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _format_single_change(
    action: str,
    event_lookup: Dict[str, CanonicalEvent],
    currency: str,
) -> str:
    parts = action.split(":")
    verb = parts[0]
    event_id = parts[1]
    event = event_lookup.get(event_id)
    raw_desc = (event.description if event and event.description else event_id).strip()
    
    # Clean description: lower case, ensure leading 'the '
    desc = raw_desc.lower()
    if not desc.startswith("the "):
        desc = f"the {desc}"
    
    if verb == "stop":
        return f"Stop {desc}"
    elif verb == "reduce_to":
        new_amt = Decimal(parts[2])
        return f"Reduce {desc} to {currency} {format_money(new_amt)}"
    return action


def format_spending_changes_clause(
    spending_changes: Tuple[str, ...],
    event_lookup: Dict[str, CanonicalEvent],
    currency: str,
) -> str:
    """Combine up to 3 spending changes into a single fluent English clause."""
    if not spending_changes:
        return ""
    
    formatted = [_format_single_change(c, event_lookup, currency) for c in spending_changes]
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        # First capitalized, second lowercased
        second = formatted[1]
        second_lower = second[0].lower() + second[1:]
        return f"{formatted[0]} and {second_lower}"
    else:
        # 3 changes
        lower_parts = [p[0].lower() + p[1:] for p in formatted[1:]]
        return f"{formatted[0]}, {lower_parts[0]}, and {lower_parts[1]}"


def compose_decision_explanation(
    decision: PolicyDecision,
    request: Dict[str, Any],
    profile: Dict[str, Any],
    amount_safe_to_pay: Decimal,
    earliest_date_for_full_payment: Optional[date],
    source_events: Optional[List[CanonicalEvent]] = None,
) -> str:
    """
    Composes the required single-sentence explanation matching ground-truth style.
    """
    cand = decision.candidate
    currency = profile.get("home_currency", "USD")
    min_keep = Decimal(str(profile.get("minimum_balance_to_keep", "0")))
    method = cand.method
    
    event_lookup = {e.event_id: e for e in (source_events or [])}
    
    if method == "full_payment":
        if cand.spending_changes:
            clause = format_spending_changes_clause(cand.spending_changes, event_lookup, currency)
            return (
                f"{clause}, then pay {currency} {format_money(cand.total_paid)} today. "
                f"This leaves at least {currency} {format_money(min_keep)} available."
            )
        else:
            return (
                f"Pay {currency} {format_money(cand.total_paid)} today. "
                f"This leaves at least {currency} {format_money(min_keep)} available over the next 90 days."
            )
            
    elif method == "installments":
        count = len(cand.payments)
        installment_amt = cand.payments[0][1] if cand.payments else Decimal("0")
        start_date = cand.payments[0][0] if cand.payments else request["request_date"]
        return (
            f"Use {count} installments of {currency} {format_money(installment_amt)}, "
            f"starting {format_date_human(start_date)}. "
            f"This leaves at least {currency} {format_money(min_keep)} available."
        )
        
    elif method == "wait":
        pay_date = cand.payments[0][0] if cand.payments else earliest_date_for_full_payment or request["request_date"]
        return (
            f"Pay {currency} {format_money(cand.total_paid)} in full on {format_date_human(pay_date)}. "
            f"Paying earlier would take the balance below the {currency} {format_money(min_keep)} minimum."
        )
        
    elif method == "partial_payment":
        pay1 = cand.payments[0] if len(cand.payments) > 0 else (request["request_date"], amount_safe_to_pay)
        pay2 = cand.payments[1] if len(cand.payments) > 1 else (earliest_date_for_full_payment, cand.total_paid - amount_safe_to_pay)
        return (
            f"Pay {currency} {format_money(pay1[1])} today and the remaining {currency} {format_money(pay2[1])} "
            f"on {format_date_human(pay2[0])}. This completes the full request and keeps the "
            f"{currency} {format_money(min_keep)} minimum protected."
        )
        
    elif method == "not_recommended":
        req_amt = Decimal(str(request["requested_amount"]))
        deadline = request["desired_completion_date"]
        if amount_safe_to_pay > Decimal("0") and (earliest_date_for_full_payment is None or earliest_date_for_full_payment > request["request_date"] + timedelta(days=90)):
            return (
                f"Do not proceed with the {currency} {format_money(req_amt)} request. "
                f"Although {currency} {format_money(amount_safe_to_pay)} is available today, "
                f"the full amount cannot be completed safely within 90 days."
            )
        else:
            return (
                f"Do not make this payment by {format_date_human(deadline)}. "
                f"None of the available options keeps the {currency} {format_money(min_keep)} minimum protected."
            )
            
    # Fallback generic
    return f"Payment of {currency} {format_money(request['requested_amount'])} is not recommended."

"""Stage 5 eligibility filters for supplied payment options.

These filters intentionally do not rank options or make a recommendation.
They provide the auditable eligibility boundary before candidate generation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Set


def accepted_payment_methods(profile: Dict[str, Any]) -> Set[str]:
    """Return the profile's explicitly accepted payment methods."""
    raw = profile.get("payment_methods_user_will_consider", "")
    return {method.strip() for method in raw.split("|") if method.strip()}


def is_method_accepted(method: str, request: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    """Check profile method preference and the request-level partial-payment flag."""
    if method not in accepted_payment_methods(profile):
        return False
    return method != "partial_payment" or bool(request.get("allows_partial_payment"))


def filter_by_deadline(option: Dict[str, Any], request: Dict[str, Any]) -> bool:
    """Return whether the literal final option payment falls by the request deadline."""
    count = option.get("number_of_payments")
    first = option.get("first_payment_date")
    frequency = option.get("payment_frequency_days")
    if count is None or first is None:
        return False
    if count == 1:
        final_date = first
    else:
        if not frequency:
            return False
        final_date = first + timedelta(days=(count - 1) * int(frequency))
    return final_date <= request["desired_completion_date"]


def filter_by_installment_cap(option: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    """Return whether an installment option respects a non-blank profile cap."""
    cap = profile.get("max_installment_months", "")
    if cap in (None, ""):
        return True
    count = option.get("number_of_payments")
    return count is not None and count <= int(cap)

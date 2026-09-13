"""
canonical.py — CanonicalEvent dataclass.

This is the single normalised representation of a financial event used by
every downstream stage.  Fields mirror ``financial_events.csv`` exactly, with
two additions:

* ``source`` — provenance of the record (explicit_event | message | image | projected).
* ``locked`` — ``True`` when a message or image has confirmed / overridden a fact.

``amount`` is stored as ``Decimal`` already converted to the user's
``home_currency`` via ``money.convert_to_home_currency``.  ``None`` means the
amount is still unresolved (e.g. waiting for image extraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal, Optional


SourceType = Literal["explicit_event", "message", "image", "projected"]


@dataclass
class CanonicalEvent:
    # ── Fields from financial_events.csv ─────────────────────────────────
    event_id: str
    user_id: str
    event_type: str                         # expense, income, investment, …
    description: str
    category: str
    direction: str                          # debit | credit
    amount: Optional[Decimal]               # home-currency Decimal; None = unresolved
    currency: str                           # original currency code
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str                             # settled, pending, scheduled, …
    linked_event_id: str                    # "" when absent
    flexibility: str                        # fixed | flexible | ""
    minimum_allowed_amount: Optional[Decimal]  # home-currency Decimal; None = N/A

    # ── Extra provenance fields ──────────────────────────────────────────
    source: SourceType = "explicit_event"
    locked: bool = False

    # ── Event-cleaning disposition (Stage 1) ─────────────────────────────
    included_in_forecast: bool = True
    exclusion_reason: Optional[str] = None

    # ── Forecast provenance (Stage 3) ────────────────────────────────────
    projected: bool = False

"""
io.py — Dataset loading, validation, and Decimal-safe CSV ingestion.

All ``amount``-like columns are read as **strings** and converted to
``Decimal`` explicitly — no silent float coercion.

Public surface:
    load_all()  → dict of DataFrames (or plain dicts) for every CSV
    load_financial_events()        → events DF + set of blank-amount event_ids
    load_financial_profiles()      → profiles DF
    load_exchange_rates()          → rates DF  (also warms the money module cache)
    load_requests()                → requests DF
    load_sample_requests()         → sample DF
    load_request_payment_options() → payment options DF
    load_messages()                → messages DF
    load_images()                  → images DF + set of missing image files
    validate_output_template()     → bool (True when output.csv header is correct)
    run_sanity_checks()            → raises AssertionError on failures
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import (
    DATASET_DIR,
    EXCHANGE_RATES_CSV,
    FINANCIAL_EVENTS_CSV,
    FINANCIAL_PROFILES_CSV,
    IMAGES_CSV,
    MEDIA_IMAGES_DIR,
    MESSAGES_CSV,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    REQUESTS_CSV,
    REQUEST_PAYMENT_OPTIONS_CSV,
    SAMPLE_REQUESTS_CSV,
)
import money as money_mod

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of ordered dicts with all values as raw strings."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _safe_decimal(val: str) -> Optional[Decimal]:
    """Convert string to Decimal; return None for blank / whitespace."""
    v = val.strip() if val else ""
    if not v:
        return None
    return Decimal(v)


def _safe_date(val: str) -> Optional[date]:
    v = val.strip() if val else ""
    if not v:
        return None
    return date.fromisoformat(v)


# ── Individual loaders ───────────────────────────────────────────────────────

def load_financial_events(
    path: Path = FINANCIAL_EVENTS_CSV,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Load financial_events.csv.

    Returns:
        events: list of row dicts with amount/minimum_allowed_amount as
                Optional[Decimal] (NOT converted to home currency yet —
                that needs profile data).  All other fields are strings.
        blank_amount_ids: set of event_id values whose original amount was
                          blank (these need later resolution from images).
    """
    raw = _read_csv(path)
    events: List[Dict[str, Any]] = []
    blank_amount_ids: Set[str] = set()

    for row in raw:
        amt = _safe_decimal(row["amount"])
        if amt is None:
            blank_amount_ids.add(row["event_id"])

        events.append({
            "event_id":               row["event_id"],
            "user_id":                row["user_id"],
            "event_type":             row["event_type"],
            "description":            row["description"],
            "category":               row["category"],
            "direction":              row["direction"],
            "amount":                 amt,                   # Decimal | None
            "currency":               row["currency"],
            "event_date":             _safe_date(row["event_date"]),
            "settlement_date":        _safe_date(row["settlement_date"]),
            "status":                 row["status"],
            "linked_event_id":        row["linked_event_id"].strip(),
            "flexibility":            row["flexibility"].strip(),
            "minimum_allowed_amount": _safe_decimal(row["minimum_allowed_amount"]),
        })

    logger.info("Loaded %d financial events (%d with blank amount).",
                len(events), len(blank_amount_ids))
    return events, blank_amount_ids


def load_financial_profiles(
    path: Path = FINANCIAL_PROFILES_CSV,
) -> List[Dict[str, Any]]:
    """Load financial_profiles.csv.  Balance fields are Decimal."""
    raw = _read_csv(path)
    profiles: List[Dict[str, Any]] = []
    for row in raw:
        profiles.append({
            "user_id":                              row["user_id"],
            "home_currency":                        row["home_currency"],
            "current_available_balance":            _safe_decimal(row["current_available_balance"]),
            "minimum_balance_to_keep":              _safe_decimal(row["minimum_balance_to_keep"]),
            "financial_priorities":                 row["financial_priorities"],
            "expense_categories_to_protect":        row["expense_categories_to_protect"],
            "expense_categories_user_is_willing_to_reduce": row["expense_categories_user_is_willing_to_reduce"],
            "expense_categories_user_is_willing_to_stop":   row["expense_categories_user_is_willing_to_stop"],
            "payment_methods_user_will_consider":   row["payment_methods_user_will_consider"],
            "max_installment_months":               row["max_installment_months"].strip(),
        })
    logger.info("Loaded %d financial profiles.", len(profiles))
    return profiles


def load_exchange_rates(path: Path = EXCHANGE_RATES_CSV) -> List[Dict[str, Any]]:
    """Load exchange_rates.csv and warm the ``money`` module cache."""
    raw = _read_csv(path)
    rates: List[Dict[str, Any]] = []
    for row in raw:
        rates.append({
            "rate_date":      _safe_date(row["rate_date"]),
            "from_currency":  row["from_currency"].strip(),
            "to_currency":    row["to_currency"].strip(),
            "rate":           Decimal(row["rate"].strip()),
        })
    # Warm the money module's internal cache from the same file
    money_mod.load_rates(path)
    logger.info("Loaded %d exchange-rate rows.", len(rates))
    return rates


def load_requests(path: Path = REQUESTS_CSV) -> List[Dict[str, Any]]:
    raw = _read_csv(path)
    requests: List[Dict[str, Any]] = []
    for row in raw:
        requests.append({
            "request_id":              row["request_id"],
            "user_id":                 row["user_id"],
            "request_date":            _safe_date(row["request_date"]),
            "request_type":            row["request_type"],
            "requested_amount":        _safe_decimal(row["requested_amount"]),
            "desired_completion_date": _safe_date(row["desired_completion_date"]),
            "allows_partial_payment":  row["allows_partial_payment"].strip().lower() == "true",
            "request_text":            row["request_text"],
        })
    logger.info("Loaded %d requests.", len(requests))
    return requests


def load_sample_requests(path: Path = SAMPLE_REQUESTS_CSV) -> List[Dict[str, Any]]:
    raw = _read_csv(path)
    samples: List[Dict[str, Any]] = []
    for row in raw:
        samples.append({
            "request_id":                  row["request_id"],
            "user_id":                     row["user_id"],
            "request_date":                _safe_date(row["request_date"]),
            "request_type":                row["request_type"],
            "requested_amount":            _safe_decimal(row["requested_amount"]),
            "desired_completion_date":     _safe_date(row["desired_completion_date"]),
            "allows_partial_payment":      row["allows_partial_payment"].strip().lower() == "true",
            "request_text":                row["request_text"],
            "amount_safe_to_pay":          row.get("amount_safe_to_pay", ""),
            "affordability_status":        row.get("affordability_status", ""),
            "recommended_payment_method":  row.get("recommended_payment_method", ""),
            "payment_plan":                row.get("payment_plan", ""),
            "earliest_date_for_full_payment": row.get("earliest_date_for_full_payment", ""),
            "spending_changes_needed":     row.get("spending_changes_needed", ""),
            "decision_explanation":        row.get("decision_explanation", ""),
        })
    logger.info("Loaded %d sample requests.", len(samples))
    return samples


def load_request_payment_options(
    path: Path = REQUEST_PAYMENT_OPTIONS_CSV,
) -> List[Dict[str, Any]]:
    raw = _read_csv(path)
    options: List[Dict[str, Any]] = []
    for row in raw:
        options.append({
            "payment_option_id":      row["payment_option_id"],
            "request_id":             row["request_id"],
            "payment_method":         row["payment_method"],
            "payment_amount":         _safe_decimal(row["payment_amount"]),
            "number_of_payments":     int(row["number_of_payments"]) if row["number_of_payments"].strip() else None,
            "first_payment_date":     _safe_date(row["first_payment_date"]),
            "payment_frequency_days": row["payment_frequency_days"].strip(),
            "financing_fee":          _safe_decimal(row["financing_fee"]),
            "total_payable_amount":   _safe_decimal(row["total_payable_amount"]),
        })
    logger.info("Loaded %d payment options.", len(options))
    return options


def load_messages(path: Path = MESSAGES_CSV) -> List[Dict[str, str]]:
    """Load messages.csv.  All values kept as strings — parsing is a later stage."""
    rows = _read_csv(path)
    logger.info("Loaded %d messages.", len(rows))
    return rows


def load_images(
    path: Path = IMAGES_CSV,
    media_dir: Path = MEDIA_IMAGES_DIR,
) -> Tuple[List[Dict[str, str]], Set[str]]:
    """
    Load images.csv and audit actual PNG files on disk.

    Returns:
        rows: list of raw row dicts.
        missing: set of image_id values whose PNG does NOT exist on disk.
    """
    rows = _read_csv(path)
    missing: Set[str] = set()
    for row in rows:
        iid = row["image_id"]
        png = media_dir / f"{iid}.png"
        if not png.exists():
            missing.add(iid)
            logger.warning("Image file missing for %s: %s", iid, png)

    logger.info("Loaded %d image rows; %d missing PNGs.", len(rows), len(missing))
    return rows, missing


# ── Output template validation ───────────────────────────────────────────────

def validate_output_template(path: Path = OUTPUT_CSV) -> bool:
    """
    Return True if ``output.csv`` has exactly the required columns in order.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    header = [h.strip() for h in header]
    if header != OUTPUT_COLUMNS:
        logger.error("output.csv header mismatch!\n  Expected: %s\n  Got:      %s",
                      OUTPUT_COLUMNS, header)
        return False
    logger.info("output.csv header matches the required schema.")
    return True


# ── Sanity checks ────────────────────────────────────────────────────────────

def run_sanity_checks() -> None:
    """
    Quick integrity assertions across the loaded data.
    Raises AssertionError with a descriptive message on failure.
    """
    requests = load_requests()
    profiles = load_financial_profiles()
    payment_opts = load_request_payment_options()
    sample_requests = load_sample_requests()
    output_rows = _read_csv(OUTPUT_CSV)

    req_ids = {r["request_id"] for r in requests}
    sample_ids = {r["request_id"] for r in sample_requests}
    all_req_ids = req_ids | sample_ids
    profile_user_ids = {p["user_id"] for p in profiles}
    opt_req_ids = {o["request_id"] for o in payment_opts}

    # 1. output.csv row count == requests.csv row count
    assert len(output_rows) == len(requests), (
        f"output.csv has {len(output_rows)} rows but requests.csv has {len(requests)}"
    )

    # 2. Every user_id in requests exists in profiles
    req_user_ids = {r["user_id"] for r in requests}
    missing_users = req_user_ids - profile_user_ids
    assert not missing_users, (
        f"Users in requests.csv missing from profiles: {missing_users}"
    )

    # 3. Every request_id in payment options exists in some request file
    orphan_opts = opt_req_ids - all_req_ids
    assert not orphan_opts, (
        f"Payment-option request_ids not found in any request file: {orphan_opts}"
    )

    logger.info("All sanity checks passed.")


# ── Convenience: load everything ─────────────────────────────────────────────

def load_all() -> Dict[str, Any]:
    """
    Load and validate all datasets.

    Returns a dict with keys:
        events, blank_amount_ids, profiles, rates, requests,
        sample_requests, payment_options, messages, images, missing_images
    """
    events, blank_ids = load_financial_events()
    profiles = load_financial_profiles()
    rates = load_exchange_rates()
    requests = load_requests()
    samples = load_sample_requests()
    options = load_request_payment_options()
    messages = load_messages()
    images, missing_images = load_images()
    validate_output_template()
    run_sanity_checks()

    return {
        "events":            events,
        "blank_amount_ids":  blank_ids,
        "profiles":          profiles,
        "rates":             rates,
        "requests":          requests,
        "sample_requests":   samples,
        "payment_options":   options,
        "messages":          messages,
        "images":            images,
        "missing_images":    missing_images,
    }

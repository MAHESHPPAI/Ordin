"""
test_stage0.py -- Comprehensive smoke + hardening tests for the four
foundational modules (config, money, canonical, data_io).

Sections
--------
A. Basic load & row-count verification              (original)
B. Blank-amount events: full manifest + predict-set breakdown
C. Image file audit
D. FX conversion end-to-end example
E. Synthetic FX unit tests: inverse fallback + nearest-prior fallback
F. Message categorisation: user-level / request / event / both
G. Sample vs request user & ID disjointness
H. Payment-options Decimal safety confirmation
I. Date-parsing + enum validation for financial_events.csv
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

import config
import money as money_mod
from money import (
    convert_to_home_currency,
    get_rate,
    load_rates,
    round_money,
    to_decimal,
    _rate_cache,
    _pair_dates,
)
import canonical
import data_io as io_mod

SEP = "=" * 72
PASS = "[PASS]"
FAIL = "[FAIL]"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record pass/fail and print result."""
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not ok:
        failures.append(name)


def main() -> None:
    print(SEP)
    print("  test_stage0 -- comprehensive foundation tests")
    print(SEP)

    # ================================================================
    # A. Basic load & row counts
    # ================================================================
    print("\n--- A. Load all datasets ---")
    data = io_mod.load_all()

    expected_counts = {
        "events":           25342,
        "profiles":         275,
        "rates":            134,
        "requests":         250,
        "sample_requests":  25,
        "payment_options":  790,
        "messages":         215,
        "images":           16,
    }
    key_map = {
        "events": "events", "profiles": "profiles", "rates": "rates",
        "requests": "requests", "sample_requests": "sample_requests",
        "payment_options": "payment_options", "messages": "messages",
        "images": "images",
    }

    print(f"\n  {'Dataset':<30} {'Expected':>8} {'Actual':>8}")
    print("  " + "-" * 50)
    for name, exp in expected_counts.items():
        actual = len(data[name])
        ok = actual == exp
        print(f"  {name:<30} {exp:>8} {actual:>8} {'  ' + PASS if ok else '  ' + FAIL}")
        if not ok:
            failures.append(f"row_count_{name}")

    # ================================================================
    # B. Blank-amount events -- full manifest + predict-set breakdown
    # ================================================================
    print("\n--- B. Blank-amount events ---")
    blanks = data["blank_amount_ids"]
    check("blank_count_is_16", len(blanks) == 16, f"got {len(blanks)}")

    # Which of these belong to users in the 250-to-predict set?
    predict_user_ids = {r["user_id"] for r in data["requests"]}
    blanks_in_predict = []
    blanks_not_predict = []
    # Build image lookup for related_event_id -> image/request
    img_lookup = {row["related_event_id"]: row for row in data["images"]}

    for ev in data["events"]:
        if ev["event_id"] in blanks:
            img = img_lookup.get(ev["event_id"], {})
            entry = {
                "event_id": ev["event_id"],
                "user_id": ev["user_id"],
                "description": ev["description"],
                "image_id": img.get("image_id", "NONE"),
                "request_id": img.get("request_id", "NONE"),
            }
            if ev["user_id"] in predict_user_ids:
                blanks_in_predict.append(entry)
            else:
                blanks_not_predict.append(entry)

    check("blanks_in_predict_set", len(blanks_in_predict) == 11,
          f"got {len(blanks_in_predict)}")
    check("blanks_outside_predict_set", len(blanks_not_predict) == 5,
          f"got {len(blanks_not_predict)}")

    print(f"\n  OCR-dependent requests ({len(blanks_in_predict)} events -> requests):")
    for b in blanks_in_predict:
        print(f"    {b['event_id']} ({b['user_id']}) -> {b['image_id']} -> {b['request_id']}")

    print(f"\n  Sample-only blank events ({len(blanks_not_predict)}):")
    for b in blanks_not_predict:
        print(f"    {b['event_id']} ({b['user_id']}) -> {b['image_id']} -> {b['request_id']}")

    # ================================================================
    # C. Image file audit
    # ================================================================
    print("\n--- C. Image file audit ---")
    missing_imgs = data["missing_images"]
    check("no_missing_images", len(missing_imgs) == 0,
          f"{len(missing_imgs)} missing" if missing_imgs else "all 16 PNGs present")

    # ================================================================
    # D. FX conversion end-to-end (event_2167: USD 1800 -> IDR)
    # ================================================================
    print("\n--- D. FX conversion example: event_2167 ---")
    amount_usd = Decimal("1800")
    settlement = date(2023, 10, 15)
    rate = get_rate("USD", "IDR", settlement)
    converted = convert_to_home_currency(amount_usd, "USD", "IDR", settlement)
    rounded = round_money(converted)

    print(f"  Amount (original) : {amount_usd} USD")
    print(f"  Settlement date   : {settlement.isoformat()}")
    print(f"  Rate USD->IDR     : {rate}")
    print(f"  Converted (full)  : {converted} IDR")
    print(f"  Rounded (output)  : {rounded} IDR")

    check("fx_rate_correct", rate == Decimal("15833.33"))
    check("fx_conversion_correct", converted == Decimal("28499994.00"))

    # ================================================================
    # E. Synthetic FX unit tests -- inverse + nearest-prior fallback
    # ================================================================
    print("\n--- E. Synthetic FX fallback tests ---")

    # Save original cache state
    orig_cache = dict(_rate_cache)
    orig_pairs = {k: list(v) for k, v in _pair_dates.items()}

    # E1. Inverse fallback: remove USD->IDR for 2023-10-15,
    #     keep IDR->USD with rate 0.0000631579 (1/15833.33 approx)
    #     Actually, let's insert a synthetic pair and remove the direct one.
    test_date = date(2099, 1, 15)

    # Insert only the reverse direction: IDR->USD = 0.00006316
    _rate_cache[("IDR", "USD", test_date)] = Decimal("0.00006316")
    _pair_dates.setdefault(("IDR", "USD"), []).append(test_date)
    _pair_dates[("IDR", "USD")].sort()

    # There should be no direct USD->IDR for 2099-01-15
    assert ("USD", "IDR", test_date) not in _rate_cache

    try:
        inv_rate = get_rate("USD", "IDR", test_date)
        expected_inv = Decimal("1") / Decimal("0.00006316")
        check("inverse_fallback_fires", True,
              f"got rate={round_money(inv_rate)}, expected ~{round_money(expected_inv)}")
        check("inverse_fallback_value", abs(inv_rate - expected_inv) < Decimal("0.01"),
              f"delta={abs(inv_rate - expected_inv)}")
    except ValueError as e:
        check("inverse_fallback_fires", False, f"raised ValueError: {e}")

    # Clean up synthetic inverse entry
    del _rate_cache[("IDR", "USD", test_date)]
    _pair_dates[("IDR", "USD")].remove(test_date)

    # E2. Nearest-prior fallback: request a date AFTER the last known date
    #     for USD->IDR. Last date in dataset with USD->IDR is some date;
    #     request one day later.
    usd_idr_dates = sorted(_pair_dates.get(("USD", "IDR"), []))
    if usd_idr_dates:
        from datetime import timedelta
        future_date = usd_idr_dates[-1] + timedelta(days=1)
        try:
            fallback_rate = get_rate("USD", "IDR", future_date)
            last_known_rate = _rate_cache[("USD", "IDR", usd_idr_dates[-1])]
            check("nearest_prior_fallback_fires", True,
                  f"requested {future_date}, fell back to {usd_idr_dates[-1]}")
            check("nearest_prior_returns_last_known_rate",
                  fallback_rate == last_known_rate,
                  f"got {fallback_rate}, expected {last_known_rate}")
        except ValueError as e:
            check("nearest_prior_fallback_fires", False, f"raised ValueError: {e}")
    else:
        check("nearest_prior_fallback_fires", False, "no USD->IDR dates found")

    # E3. Verify ValueError when no rate exists at all
    try:
        get_rate("XYZ", "ABC", date(2025, 1, 1))
        check("missing_pair_raises", False, "should have raised ValueError")
    except ValueError:
        check("missing_pair_raises", True, "ValueError raised correctly")

    # E4. Same-currency returns 1
    check("same_currency_rate_is_1",
          get_rate("USD", "USD", date(2025, 1, 1)) == Decimal("1"))

    # ================================================================
    # F. Message categorisation
    # ================================================================
    print("\n--- F. Message categorisation ---")
    msgs = data["messages"]
    user_only = [m for m in msgs if m["request_id"].strip() == "" and m["related_event_id"].strip() == ""]
    req_only = [m for m in msgs if m["request_id"].strip() != "" and m["related_event_id"].strip() == ""]
    evt_only = [m for m in msgs if m["request_id"].strip() == "" and m["related_event_id"].strip() != ""]
    both = [m for m in msgs if m["request_id"].strip() != "" and m["related_event_id"].strip() != ""]

    print(f"  Total messages: {len(msgs)}")
    print(f"    User-level only (no req, no event): {len(user_only)}")
    print(f"    Request-linked only:                {len(req_only)}")
    print(f"    Event-linked only:                  {len(evt_only)}")
    print(f"    Both req + event linked:            {len(both)}")

    check("user_level_msgs_76", len(user_only) == 76)
    check("request_linked_msgs_100", len(req_only) == 100)
    check("event_linked_msgs_11", len(evt_only) == 11)
    check("both_linked_msgs_28", len(both) == 28)
    check("msg_categories_sum", len(user_only) + len(req_only) + len(evt_only) + len(both) == len(msgs))

    # ================================================================
    # G. Sample vs request user/ID disjointness
    # ================================================================
    print("\n--- G. Sample vs request disjointness ---")
    req_user_ids = {r["user_id"] for r in data["requests"]}
    sample_user_ids = {r["user_id"] for r in data["sample_requests"]}
    user_overlap = req_user_ids & sample_user_ids
    check("users_fully_disjoint", len(user_overlap) == 0,
          f"overlap: {user_overlap}" if user_overlap else "zero overlap")

    req_ids = {r["request_id"] for r in data["requests"]}
    sample_req_ids = {r["request_id"] for r in data["sample_requests"]}
    id_overlap = req_ids & sample_req_ids
    check("request_ids_fully_disjoint", len(id_overlap) == 0,
          f"overlap: {id_overlap}" if id_overlap else "zero overlap")

    # ================================================================
    # H. Payment-options Decimal safety
    # ================================================================
    print("\n--- H. Payment-options Decimal safety ---")
    opts = data["payment_options"]

    # Check that amount fields are Decimal, not float
    sample_opt = opts[0]
    check("payment_amount_is_decimal",
          isinstance(sample_opt["payment_amount"], Decimal),
          f"type={type(sample_opt['payment_amount']).__name__}")
    check("financing_fee_is_decimal",
          isinstance(sample_opt["financing_fee"], Decimal),
          f"type={type(sample_opt['financing_fee']).__name__}")
    check("total_payable_is_decimal",
          isinstance(sample_opt["total_payable_amount"], Decimal),
          f"type={type(sample_opt['total_payable_amount']).__name__}")

    # Confirm zero blanks in financing_fee
    blank_fees = sum(1 for o in opts if o["financing_fee"] is None)
    check("no_blank_financing_fees", blank_fees == 0, f"blanks: {blank_fees}")

    # ================================================================
    # I. Date-parsing + enum validation for financial_events.csv
    # ================================================================
    print("\n--- I. Event enum + date validation ---")
    events = data["events"]

    bad_directions = [e["event_id"] for e in events if e["direction"] not in config.EVENT_DIRECTIONS]
    bad_statuses = [e["event_id"] for e in events if e["status"] not in config.EVENT_STATUSES]
    bad_types = [e["event_id"] for e in events if e["event_type"] not in config.EVENT_TYPES]
    bad_flex = [e["event_id"] for e in events if e["flexibility"] and e["flexibility"] not in config.EVENT_FLEXIBILITIES]

    check("all_directions_valid", len(bad_directions) == 0,
          f"{len(bad_directions)} bad: {bad_directions[:5]}" if bad_directions else "all valid")
    check("all_statuses_valid", len(bad_statuses) == 0,
          f"{len(bad_statuses)} bad: {bad_statuses[:5]}" if bad_statuses else "all valid")
    check("all_event_types_valid", len(bad_types) == 0,
          f"{len(bad_types)} bad: {bad_types[:5]}" if bad_types else "all valid")
    check("all_flexibilities_valid", len(bad_flex) == 0,
          f"{len(bad_flex)} bad: {bad_flex[:5]}" if bad_flex else "all valid")

    # Date parsing -- all event_date and settlement_date should be valid dates or None
    bad_dates = []
    for e in events:
        for col_name in ["event_date", "settlement_date"]:
            val = e[col_name]
            if val is not None and not isinstance(val, date):
                bad_dates.append((e["event_id"], col_name, val))
    check("all_dates_parse_cleanly", len(bad_dates) == 0,
          f"{len(bad_dates)} bad" if bad_dates else "all valid")

    # CanonicalEvent import check
    print("\n--- Canonical dataclass ---")
    from canonical import CanonicalEvent
    demo = CanonicalEvent(
        event_id="event_2167", user_id="user_25", event_type="income",
        description="USD freelance income", category="freelance_income",
        direction="credit", amount=converted, currency="USD",
        event_date=settlement, settlement_date=settlement,
        status="settled", linked_event_id="", flexibility="fixed",
        minimum_allowed_amount=None, source="explicit_event", locked=False,
    )
    check("canonical_creates_ok", demo.event_id == "event_2167")
    check("canonical_amount_is_decimal", isinstance(demo.amount, Decimal))

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{SEP}")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        print(SEP)
        sys.exit(1)
    else:
        print("  All checks passed [OK]")
        print(SEP)


if __name__ == "__main__":
    main()

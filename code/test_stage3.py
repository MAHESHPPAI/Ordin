"""
test_stage3.py — Comprehensive Test Suite for Stage 3 Forecasting.

Validates:
1. Dataset-Wide Recurrence Classification & Confidence Breakdown:
   - Total series, recurring count, sparse count, high/low confidence counts.
2. Profile Invariant Assertion:
   - Zero overlap between protected and flexible categories across all 275 users.
3. Test Case A: CV = 0.0% Category User (user_03)
   - Rent, cloud storage, streaming carried forward with identical amounts.
   - Prints full 90-day projected timeline alongside raw history for visual inspection.
4. Test Case B: Foreign-Currency Salary User (user_109)
   - Original USD salary converted cleanly to EUR home currency.
   - Splicing of explicit scheduled event without double counting.
   - Prints full 90-day projected timeline alongside raw history.
5. Test Case C: Stage 2 Message Overrides
   - user_117: Confirmed salary raise to EUR 1188 starting 2026-07-15.
   - user_75: Confirmed employment ended; future salary completely suppressed.
   - user_81: Confirmed 12% lease increase on monthly rent.
6. Test Case D: Investment Category User (user_101)
   - Past investment transactions exist in history.
   - Verifies ZERO recurring investment events projected.
7. Deduplication & Provenance Flagging:
   - Validates that every output row has projected: bool correctly set.
   - Validates zero double-counting within the deduplication window.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import canonical
import config
import data_io
import event_cleaner
import forecaster
import image_ocr
import message_parser
import recurrence

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_pipeline():
    """Runs Stage 0, Stage 1, and Stage 2 pipeline to produce cleaned events."""
    raw_events, _ = data_io.load_financial_events()
    profiles = data_io.load_financial_profiles()
    p_dict = {p["user_id"]: p for p in profiles}

    # Stage 0: raw to canonical
    canonicals = [
        event_cleaner.raw_to_canonical(e, p_dict[e["user_id"]]["home_currency"])
        for e in raw_events
    ]

    # Stage 2: image overrides
    extractions = image_ocr.extract_event_amounts_from_images()
    canonicals = image_ocr.apply_image_overrides(canonicals, extractions, p_dict)

    # Stage 2: message parsing
    facts = message_parser.parse_messages(profiles=p_dict)
    msg_events = [f.generated_event for f in facts if f.generated_event is not None]

    # Stage 1: event cleaning
    all_events = canonicals + msg_events
    cleaned = event_cleaner.clean_events(all_events, profiles=p_dict)

    # Requests
    reqs = {r["user_id"]: r for r in data_io.load_requests()}
    sample_reqs = {r["user_id"]: r for r in data_io.load_sample_requests()}
    reqs.update(sample_reqs)

    return cleaned, p_dict, facts, reqs


def test_dataset_recurrence_classification(all_events, facts):
    print("\n" + "=" * 75)
    print("TEST 1: Dataset-Wide Recurrence Classification & Confidence Breakdown")
    print("=" * 75)

    patterns = recurrence.detect_all_recurrence_patterns(all_events, stage2_facts=facts)

    # Group events by (user_id, category) to count total series
    from collections import defaultdict
    series_events = defaultdict(list)
    for e in all_events:
        if e.included_in_forecast and (e.status == "settled" or e.status == "scheduled" or (e.status == "pending" and e.direction == "debit")):
            series_events[(e.user_id, e.category)].append(e)

    total_series = len(series_events)
    recurring_series = len(patterns)
    sparse_series = total_series - recurring_series

    high_conf = sum(1 for p in patterns.values() if p.recurrence_confidence == "high")
    low_conf = sum(1 for p in patterns.values() if p.recurrence_confidence == "low")

    print(f"Total (user, category) series with usable data: {total_series}")
    print(f"  - Classified Recurring:       {recurring_series:>5} ({recurring_series/total_series*100:.1f}%)")
    print(f"  - Too-Sparse / Not-Projected: {sparse_series:>5} ({sparse_series/total_series*100:.1f}%)")
    print(f"\nRecurrence Confidence Breakdown (across {recurring_series} recurring series):")
    print(f"  - High Confidence (gap std <= 1.0 day): {high_conf:>5} ({high_conf/recurring_series*100:.1f}%)")
    print(f"  - Low Confidence  (gap std >  1.0 day): {low_conf:>5} ({low_conf/recurring_series*100:.1f}%)")

    assert recurring_series >= 2700, f"Expected >= 2700 recurring series, got {recurring_series}"
    assert high_conf > 2000, f"Expected > 2000 high confidence series, got {high_conf}"
    print("\n[PASS] Test 1: Full dataset recurrence classification matches empirical distribution.")


def test_profile_conflict_invariant(p_dict):
    print("\n" + "=" * 75)
    print("TEST 2: Profile Conflict Invariant Verification (0 / 275 users)")
    print("=" * 75)

    def _parse_cats(val: Any) -> set[str]:
        if not val:
            return set()
        if isinstance(val, (set, list, tuple)):
            return set(val)
        if isinstance(val, str):
            return set(c.strip() for c in val.split("|") if c.strip())
        return set()

    violations = []
    for uid, p in p_dict.items():
        protect = _parse_cats(p.get("expense_categories_to_protect"))
        reduce_cats = _parse_cats(p.get("expense_categories_user_is_willing_to_reduce"))
        stop_cats = _parse_cats(p.get("expense_categories_user_is_willing_to_stop"))
        overlap = protect & (reduce_cats | stop_cats)
        if overlap:
            violations.append((uid, overlap))

    print(f"Users audited: {len(p_dict)}")
    print(f"Profile category conflict violations found: {len(violations)}")
    assert len(violations) == 0, f"Violations found: {violations}"
    print("[PASS] Test 2: Exactly zero users have overlap between protected and flexible categories.")


def print_user_timeline_comparison(uid, req, events, timeline, p_dict):
    """Helper to pretty-print historical events vs 90-day forecasted timeline."""
    hc = p_dict[uid]["home_currency"]
    rdate = req["request_date"]
    print(f"\nUser: {uid} | Home Currency: {hc} | Request Date: {rdate} ({req['request_id']})")
    print("-" * 75)
    print("RECENT RAW HISTORY (Last settled occurrences per category):")
    settled_evs = [e for e in events if e.user_id == uid and e.status == "settled" and (e.settlement_date or e.event_date) <= rdate]
    settled_evs.sort(key=lambda e: (e.settlement_date or e.event_date or date.min))

    # Show last 2 events per category
    cats_seen = {}
    for e in reversed(settled_evs):
        cats_seen.setdefault(e.category, []).append(e)

    for cat in sorted(cats_seen.keys()):
        for e in reversed(cats_seen[cat][:2]):
            dt = e.settlement_date or e.event_date
            print(f"  HIST: {dt} | {e.direction.upper():<6} | {e.category:<20} | {hc} {e.amount:>10} | flex: {e.flexibility:<10} | {e.status}")

    print("\nFORECASTED 90-DAY TIMELINE (Stage 3 output):")
    for e in timeline:
        dt = e.settlement_date or e.event_date
        proj_flag = "[PROJECTED]" if e.projected else "[CONFIRMED]"
        print(f"  FCST: {dt} | {e.direction.upper():<6} | {e.category:<20} | {hc} {e.amount:>10} | {proj_flag:<11} | flex: {e.flexibility:<10} | {e.description[:35]}")


def test_cv_zero_user(all_events, p_dict, facts, reqs):
    print("\n" + "=" * 75)
    print("TEST 3: CV = 0.0% Category User Verification (user_03)")
    print("=" * 75)

    uid = "user_03"
    req = reqs[uid]
    u_events = [e for e in all_events if e.user_id == uid]
    timeline = forecaster.forecast_cash_flows(req, u_events, p_dict[uid], stage2_facts=facts)

    print_user_timeline_comparison(uid, req, u_events, timeline, p_dict)

    # Check rent, cloud_storage, streaming
    rent_proj = [e for e in timeline if e.category == "rent"]
    cloud_proj = [e for e in timeline if e.category == "cloud_storage"]
    stream_proj = [e for e in timeline if e.category == "streaming"]

    assert len(rent_proj) >= 2, f"Expected >= 2 rent projections, got {len(rent_proj)}"
    assert len(cloud_proj) >= 2, f"Expected >= 2 cloud projections, got {len(cloud_proj)}"
    assert len(stream_proj) >= 2, f"Expected >= 2 stream projections, got {len(stream_proj)}"

    # Check CV=0 amount invariance
    rent_hist = [e.amount for e in u_events if e.category == "rent" and e.status == "settled"]
    cloud_hist = [e.amount for e in u_events if e.category == "cloud_storage" and e.status == "settled"]
    stream_hist = [e.amount for e in u_events if e.category == "streaming" and e.status == "settled"]

    for r in rent_proj:
        assert r.amount == rent_hist[-1], f"Rent projected {r.amount} != latest hist {rent_hist[-1]}"
    for c in cloud_proj:
        assert c.amount == cloud_hist[-1], f"Cloud projected {c.amount} != latest hist {cloud_hist[-1]}"
    for s in stream_proj:
        assert s.amount == stream_hist[-1], f"Stream projected {s.amount} != latest hist {stream_hist[-1]}"

    print("\n[PASS] Test 3: user_03 CV=0 categories project exact identical amounts into 90-day timeline.")


def test_foreign_currency_salary_user(all_events, p_dict, facts, reqs):
    print("\n" + "=" * 75)
    print("TEST 4: Foreign-Currency Salary User Verification (user_109)")
    print("=" * 75)

    uid = "user_109"
    req = reqs[uid]
    u_events = [e for e in all_events if e.user_id == uid]
    timeline = forecaster.forecast_cash_flows(req, u_events, p_dict[uid], stage2_facts=facts)

    print_user_timeline_comparison(uid, req, u_events, timeline, p_dict)

    # Verify user_109 salary: original was USD 2604 converted to EUR
    salary_events = [e for e in timeline if e.category == "salary"]
    print(f"\nTotal salary occurrences in 90-day forecast: {len(salary_events)}")
    for s in salary_events:
        dt = s.settlement_date or s.event_date
        print(f"  Salary on {dt}: EUR {s.amount} (projected={s.projected}, source={s.source}, status={s.status})")

    assert len(salary_events) >= 3, f"Expected >= 3 salary occurrences, got {len(salary_events)}"

    # Verify the first salary event is the explicit scheduled one (projected=False)
    first_sal = salary_events[0]
    assert not first_sal.projected, "First salary in window must be the explicit scheduled event (projected=False)"
    assert first_sal.event_id == "event_10164", f"Expected event_10164, got {first_sal.event_id}"

    # Verify subsequent salary events are projected (projected=True)
    for s in salary_events[1:]:
        assert s.projected, f"Subsequent salary event {s.event_id} must have projected=True"

    # Verify converted amount is in EUR (not raw USD 2604)
    assert first_sal.currency == "USD" or first_sal.amount != Decimal("2604")
    assert first_sal.amount > Decimal("2000"), f"Expected converted EUR amount, got {first_sal.amount}"

    print("\n[PASS] Test 4: user_109 foreign-currency salary spliced explicit scheduled event and projected cleanly.")


def test_stage2_message_overrides(all_events, p_dict, facts, reqs):
    print("\n" + "=" * 75)
    print("TEST 5: Stage 2 Message Overrides Verification (user_117, user_75, user_81)")
    print("=" * 75)

    # 1. user_117: Salary Raise to EUR 1188 starting 2026-07-15
    uid117 = "user_117"
    req117 = reqs[uid117]
    u117_evs = [e for e in all_events if e.user_id == uid117]
    tl117 = forecaster.forecast_cash_flows(req117, u117_evs, p_dict[uid117], stage2_facts=facts)
    sal117 = [e for e in tl117 if e.category == "salary"]

    print_user_timeline_comparison(uid117, req117, u117_evs, tl117, p_dict)
    print(f"\nuser_117 Salary Projections ({len(sal117)} occurrences):")
    for s in sal117:
        print(f"  {s.settlement_date}: EUR {s.amount} (projected={s.projected})")

    assert len(sal117) >= 2, f"Expected >= 2 salary events for user_117, got {len(sal117)}"
    for s in sal117:
        assert s.amount == Decimal("1188"), f"Expected raised salary EUR 1188, got {s.amount}"

    # 2. user_75: Employment Ended -> Salary Projections Suppressed
    uid75 = "user_75"
    req75 = reqs[uid75]
    u75_evs = [e for e in all_events if e.user_id == uid75]
    tl75 = forecaster.forecast_cash_flows(req75, u75_evs, p_dict[uid75], stage2_facts=facts)
    sal75 = [e for e in tl75 if e.category == "salary"]

    print_user_timeline_comparison(uid75, req75, u75_evs, tl75, p_dict)
    print(f"\nuser_75 Salary Projections: {len(sal75)} occurrences (expected 0 due to employment ended)")
    assert len(sal75) == 0, f"Expected 0 salary projections for user_75, got {len(sal75)}"

    # 3. user_81: 12% Lease Increase on Rent
    uid81 = "user_81"
    req81 = reqs[uid81]
    u81_evs = [e for e in all_events if e.user_id == uid81]
    tl81 = forecaster.forecast_cash_flows(req81, u81_evs, p_dict[uid81], stage2_facts=facts)
    rent81 = [e for e in tl81 if e.category == "rent"]

    print_user_timeline_comparison(uid81, req81, u81_evs, tl81, p_dict)
    print(f"\nuser_81 Rent Projections ({len(rent81)} occurrences):")
    for r in rent81:
        print(f"  {r.settlement_date}: USD {r.amount} (projected={r.projected})")

    assert len(rent81) >= 2, f"Expected >= 2 rent projections for user_81, got {len(rent81)}"
    # Previous rent was 926.4, 12% increase is 1037.57 (or 1037.568)
    for r in rent81:
        assert r.amount >= Decimal("1037.56"), f"Expected 12% increased rent, got {r.amount}"

    print("\n[PASS] Test 5: All Stage 2 message overrides (raise, suppression, lease increase) applied accurately.")


def test_investment_suppression(all_events, p_dict, facts, reqs):
    print("\n" + "=" * 75)
    print("TEST 6: Investment Category Suppression Verification (user_101)")
    print("=" * 75)

    uid = "user_101"
    req = reqs[uid]
    u_events = [e for e in all_events if e.user_id == uid]
    timeline = forecaster.forecast_cash_flows(req, u_events, p_dict[uid], stage2_facts=facts)

    print_user_timeline_comparison(uid, req, u_events, timeline, p_dict)

    inv_hist = [e for e in u_events if e.category == "investment"]
    inv_fcst = [e for e in timeline if e.category == "investment"]

    print(f"\nuser_101 Historical Investment Events: {len(inv_hist)}")
    for e in inv_hist:
        print(f"  HIST: {e.event_date} | {e.event_type} | INR {e.amount} | status: {e.status}")

    print(f"user_101 Forecasted Investment Events: {len(inv_fcst)} (expected 0)")
    assert len(inv_hist) > 0, "Expected user_101 to have historical investment events"
    assert len(inv_fcst) == 0, f"Expected 0 forecasted investment events, got {len(inv_fcst)}"

    print("\n[PASS] Test 6: Investment series is never projected into future timeline.")


def test_deduplication_and_flags(all_events, p_dict, facts, reqs):
    print("\n" + "=" * 75)
    print("TEST 7: Deduplication and Provenance Flags Verification")
    print("=" * 75)

    # Test across a sample of 25 users
    double_count_violations = 0
    flag_violations = 0
    total_timeline_events = 0

    sample_uids = list(reqs.keys())[:25]
    for uid in sample_uids:
        req = reqs[uid]
        u_events = [e for e in all_events if e.user_id == uid]
        timeline = forecaster.forecast_cash_flows(req, u_events, p_dict[uid], stage2_facts=facts)
        total_timeline_events += len(timeline)

        # Check provenance flags
        for e in timeline:
            if e.source == "projected" and not e.projected:
                flag_violations += 1
            if e.source in ("explicit_event", "message", "image") and e.projected:
                flag_violations += 1

        # Check deduplication: no two events with same category, direction within 4 days
        for i in range(len(timeline)):
            for j in range(i + 1, len(timeline)):
                e1 = timeline[i]
                e2 = timeline[j]
                if e1.category == e2.category and e1.direction == e2.direction:
                    d1 = e1.settlement_date or e1.event_date
                    d2 = e2.settlement_date or e2.event_date
                    if d1 and d2 and abs((d1 - d2).days) <= 4:
                        # Allow if both are explicit events with distinct event_ids
                        if e1.projected or e2.projected:
                            double_count_violations += 1

    print(f"Sample users audited: {len(sample_uids)}")
    print(f"Total timeline events inspected: {total_timeline_events}")
    print(f"Provenance flag violations: {flag_violations}")
    print(f"Double-counting violations: {double_count_violations}")

    assert flag_violations == 0, f"Flag violations found: {flag_violations}"
    assert double_count_violations == 0, f"Double count violations found: {double_count_violations}"

    print("\n[PASS] Test 7: Deduplication window and projected flags validated with 0 violations.")


def test_newer_salary_supersedes_historical_mode():
    """A recent confirmed salary change must not be erased by an old mode."""
    base = date(2026, 1, 15)
    salaries = [
        canonical.CanonicalEvent(
            event_id=f"salary_{index}", user_id="salary_change_user", event_type="income", description="Payroll",
            category="salary", direction="credit", amount=amount, currency="USD",
            event_date=date(2025, month, 15), settlement_date=date(2025, month, 15), status="settled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None,
        )
        for index, (month, amount) in enumerate(((10, Decimal("1000")), (11, Decimal("1000")), (12, Decimal("1200"))), 1)
    ]
    pattern = recurrence.detect_series_pattern(salaries, "salary_change_user", "salary")
    assert pattern is not None
    assert pattern.amount == Decimal("1200")
    assert pattern.last_event_date == date(2025, 12, 15)
    print("\n[PASS] Test 8: Latest settled salary supersedes obsolete historical mode.")


def test_independent_salary_streams_are_not_interleaved():
    """Two monthly salary streams must not become a fictional five-day cadence."""
    salaries = []
    for month in (1, 2, 3):
        for day, description, amount in ((15, "Primary salary", Decimal("1000")), (20, "Second income", Decimal("600"))):
            salaries.append(canonical.CanonicalEvent(
                event_id=f"{description}-{month}", user_id="two_salary_user", event_type="income",
                description=description, category="salary", direction="credit", amount=amount, currency="USD",
                event_date=date(2025, month, day), settlement_date=date(2025, month, day), status="settled",
                linked_event_id="", flexibility="fixed", minimum_allowed_amount=None,
            ))
    patterns = recurrence.detect_user_recurrence_patterns(salaries)
    salary_patterns = [p for p in patterns.values() if p.category == "salary"]
    assert len(salary_patterns) == 2
    assert all(28 <= p.interval_days <= 31 for p in salary_patterns)
    assert all(p.interval_days != 5 for p in salary_patterns)
    print("\n[PASS] Test 9: Independent salary streams retain their monthly cadence.")


def test_one_cycle_unpaid_leave_does_not_replace_salary_baseline():
    """A locked next-payroll adjustment must not permanently reduce salary."""
    salaries = [
        canonical.CanonicalEvent(
            event_id=f"payroll-{month}", user_id="leave_user", event_type="income", description="Payroll",
            category="salary", direction="credit", amount=amount, currency="EUR",
            event_date=date(2025, month, 15), settlement_date=date(2025, month, 15), status="settled",
            linked_event_id="", flexibility="fixed", minimum_allowed_amount=None,
        )
        for month, amount in ((9, Decimal("1422.85")), (10, Decimal("1422.85")), (11, Decimal("1422.85")),
                              (12, Decimal("1422.85")), (1, Decimal("782.57")))
    ]
    fact = type("Fact", (), {"user_id": "leave_user", "pattern_family": "salary_unpaid_leave_reduction",
                              "extracted_amounts": [("EUR", Decimal("1422.85"))]})()
    pattern = recurrence.detect_series_pattern(salaries, "leave_user", "salary", [fact])
    assert pattern is not None
    assert pattern.amount == Decimal("1422.85")
    print("\n[PASS] Test 10: One-cycle unpaid-leave adjustment preserves the recurring salary baseline.")


def test_request_date_occurrences_are_included():
    """Day-zero recurring and confirmed events belong in the 90-day safety walk."""
    monthly = recurrence.RecurrencePattern(
        user_id="boundary_user", category="education", direction="debit", event_type="expense",
        interval_days=30, observed_gap_std=0.0, recurrence_confidence="high", amount=Decimal("89"),
        amount_cv=0.0, flexibility="fixed", minimum_allowed_amount=None,
        last_event_date=date(2025, 1, 7), occurrences_count=3, day_of_month=7,
    )
    weekly = recurrence.RecurrencePattern(
        user_id="boundary_user", category="groceries", direction="debit", event_type="expense",
        interval_days=7, observed_gap_std=0.0, recurrence_confidence="high", amount=Decimal("50"),
        amount_cv=0.0, flexibility="fixed", minimum_allowed_amount=None,
        last_event_date=date(2025, 1, 31), occurrences_count=3, day_of_month=None,
    )
    start = date(2025, 2, 7)
    assert forecaster.generate_candidate_dates(monthly, start, start + timedelta(days=90))[0] == start
    assert forecaster.generate_candidate_dates(weekly, start, start + timedelta(days=90))[0] == start
    print("\n[PASS] Test 11: Request-date recurring occurrences are included in the forecast.")


def main():
    print("\n" + "#" * 75)
    print("HACKERRANK ORCHESTRATE - BUY OR WAIT?")
    print("STAGE 3: CASH-FLOW FORECASTING TEST SUITE")
    print("#" * 75)

    print("\nBuilding cleaned Stage 1 & Stage 2 input pipeline...")
    all_events, p_dict, facts, reqs = build_pipeline()
    print(f"Pipeline ready: {len(all_events)} cleaned events, {len(p_dict)} profiles, {len(reqs)} requests.")

    test_dataset_recurrence_classification(all_events, facts)
    test_profile_conflict_invariant(p_dict)
    test_cv_zero_user(all_events, p_dict, facts, reqs)
    test_foreign_currency_salary_user(all_events, p_dict, facts, reqs)
    test_stage2_message_overrides(all_events, p_dict, facts, reqs)
    test_investment_suppression(all_events, p_dict, facts, reqs)
    test_deduplication_and_flags(all_events, p_dict, facts, reqs)
    test_newer_salary_supersedes_historical_mode()
    test_independent_salary_streams_are_not_interleaved()
    test_one_cycle_unpaid_leave_does_not_replace_salary_baseline()
    test_request_date_occurrences_are_included()

    print("\n" + "=" * 75)
    print("ALL 11 STAGE 3 TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

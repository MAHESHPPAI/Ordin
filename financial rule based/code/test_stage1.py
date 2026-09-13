"""
test_stage1.py -- Comprehensive tests for event_cleaner.py (Stage 1).

Sections:
  A. Full pipeline run: counts per exclusion reason
  B. Status filter tests: independent verification of each filter
  C. Linked-event resolution: all 58 rows, grouped by pattern
  D. Unrecognised-pattern detection
  E. Flexibility / minimum_allowed_amount invariant check
  F. Balance-authority boundary assertion
  G. Conflict-resolution utility test
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

import config
import data_io
import event_cleaner
from event_cleaner import (
    apply_status_filters,
    clean_events,
    filter_cancelled,
    filter_failed,
    filter_pending_credit,
    filter_unrealized_investment,
    resolve_all_linked_events,
    resolve_conflict,
    resolve_pending_settled,
    resolve_scheduled_failed,
    resolve_settled_cancelled,
    resolve_settled_settled,
    resolve_unrealized_settled,
    validate_flexibility,
    raw_to_canonical,
    BALANCE_AUTHORITY_NOTE,
)
from canonical import CanonicalEvent

SEP = "=" * 72
PASS = "[PASS]"
FAIL = "[FAIL]"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not ok:
        failures.append(name)


def main() -> None:
    print(SEP)
    print("  test_stage1 -- event cleaning tests")
    print(SEP)

    # Load raw data
    raw_events, blank_ids = data_io.load_financial_events()
    profiles = data_io.load_financial_profiles()
    data_io.load_exchange_rates()  # warm FX cache

    total_raw = len(raw_events)
    check("raw_event_count", total_raw == 25342, f"got {total_raw}")

    # ================================================================
    # A. Full pipeline run
    # ================================================================
    print("\n--- A. Full pipeline: clean all events ---")
    all_events = clean_events(raw_events, profiles)
    check("output_count_equals_input", len(all_events) == total_raw,
          f"in={total_raw}, out={len(all_events)}")

    included = [ev for ev in all_events if ev.included_in_forecast]
    excluded = [ev for ev in all_events if not ev.included_in_forecast]
    print(f"\n  Total:    {len(all_events)}")
    print(f"  Included: {len(included)}")
    print(f"  Excluded: {len(excluded)}")

    # Counts by exclusion reason
    reason_counts = Counter(ev.exclusion_reason for ev in excluded)
    print(f"\n  Exclusion reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    # Expected: cancelled=22, failed=21, pending_credit=8, unrealized=10
    # Plus linkage-based exclusions (some overlap with status filters)
    check("cancelled_excluded", reason_counts.get("cancelled_transaction", 0) == 22,
          f"got {reason_counts.get('cancelled_transaction', 0)}")
    check("failed_excluded", reason_counts.get("failed_transaction", 0) == 21,
          f"got {reason_counts.get('failed_transaction', 0)}")
    check("pending_credit_excluded", reason_counts.get("pending_credit", 0) == 8,
          f"got {reason_counts.get('pending_credit', 0)}")
    check("unrealized_excluded", reason_counts.get("unrealized_investment", 0) == 10,
          f"got {reason_counts.get('unrealized_investment', 0)}")

    # Pending debits should NOT be excluded
    pending_debits = [ev for ev in all_events
                      if ev.status == "pending" and ev.direction == "debit"]
    check("pending_debits_all_included",
          all(ev.included_in_forecast for ev in pending_debits),
          f"{len(pending_debits)} pending debits, "
          f"{sum(1 for ev in pending_debits if ev.included_in_forecast)} included")

    # Scheduled events should all be included
    scheduled = [ev for ev in all_events if ev.status == "scheduled"]
    check("scheduled_all_included",
          all(ev.included_in_forecast for ev in scheduled),
          f"{len(scheduled)} scheduled events")

    # Settled events should all be included (unless none break)
    settled = [ev for ev in all_events if ev.status == "settled"]
    check("settled_all_included",
          all(ev.included_in_forecast for ev in settled),
          f"{len(settled)} settled events")

    # ================================================================
    # B. Independent status filter tests
    # ================================================================
    print("\n--- B. Independent status filter tests ---")

    # Create test events for each case
    def make_test_event(**kwargs) -> CanonicalEvent:
        defaults = dict(
            event_id="test", user_id="u", event_type="expense",
            description="test", category="test", direction="debit",
            amount=Decimal("100"), currency="INR",
            event_date=date(2025, 1, 1), settlement_date=date(2025, 1, 1),
            status="settled", linked_event_id="", flexibility="fixed",
            minimum_allowed_amount=None,
        )
        defaults.update(kwargs)
        return CanonicalEvent(**defaults)

    # Cancelled
    ev_cancelled = make_test_event(status="cancelled")
    check("filter_cancelled_excludes", filter_cancelled(ev_cancelled) == "cancelled_transaction")
    check("filter_cancelled_keeps_settled", filter_cancelled(make_test_event(status="settled")) is None)

    # Failed
    ev_failed = make_test_event(status="failed")
    check("filter_failed_excludes", filter_failed(ev_failed) == "failed_transaction")
    check("filter_failed_keeps_pending", filter_failed(make_test_event(status="pending")) is None)

    # Pending credit vs debit
    ev_pending_credit = make_test_event(status="pending", direction="credit")
    ev_pending_debit = make_test_event(status="pending", direction="debit")
    check("filter_pending_credit_excludes_credit",
          filter_pending_credit(ev_pending_credit) == "pending_credit")
    check("filter_pending_credit_keeps_debit",
          filter_pending_credit(ev_pending_debit) is None)

    # Unrealized
    ev_unrealized = make_test_event(status="unrealized", direction="non_cash")
    check("filter_unrealized_excludes",
          filter_unrealized_investment(ev_unrealized) == "unrealized_investment")
    check("filter_unrealized_keeps_settled",
          filter_unrealized_investment(make_test_event(status="settled")) is None)

    # Combined filter: pending debit should pass ALL filters
    check("combined_pending_debit_passes",
          apply_status_filters(ev_pending_debit) is None)
    # Combined: cancelled should be caught
    check("combined_cancelled_caught",
          apply_status_filters(ev_cancelled) is not None)

    # ================================================================
    # C. Linked-event resolution: all 58 rows grouped by pattern
    # ================================================================
    print("\n--- C. Linked-event resolution ---")

    events_by_id = {ev.event_id: ev for ev in all_events}
    linked_events = [ev for ev in all_events if ev.linked_event_id]
    check("linked_event_count", len(linked_events) == 58, f"got {len(linked_events)}")

    # Group by pattern
    pattern_groups: dict[str, list[tuple[CanonicalEvent, CanonicalEvent]]] = defaultdict(list)
    orphans = []
    for ev in linked_events:
        parent = events_by_id.get(ev.linked_event_id)
        if parent is None:
            orphans.append(ev.event_id)
            continue
        pattern = f"{ev.status}->{parent.status}"
        pattern_groups[pattern].append((ev, parent))

    check("no_orphan_links", len(orphans) == 0,
          f"orphans: {orphans}" if orphans else "none")

    expected_patterns = {
        "settled->settled": 19,
        "pending->settled": 14,
        "unrealized->settled": 10,
        "settled->cancelled": 8,
        "scheduled->failed": 7,
    }

    print(f"\n  Pattern breakdown:")
    for pattern, pairs in sorted(pattern_groups.items()):
        expected = expected_patterns.get(pattern, "???")
        print(f"    {pattern}: {len(pairs)} (expected {expected})")
        check(f"pattern_{pattern}_count", len(pairs) == expected,
              f"got {len(pairs)}, expected {expected}")

        # Print first 2 samples
        for child, parent in pairs[:2]:
            print(f"      {child.event_id} ({child.event_type}/{child.direction}) "
                  f"-> {parent.event_id} ({parent.event_type}/{parent.direction})")

    # ================================================================
    # D. Unrecognised pattern detection
    # ================================================================
    print("\n--- D. Unrecognised pattern detection ---")
    _, unrecognised = resolve_all_linked_events(events_by_id)
    check("no_unrecognised_patterns", len(unrecognised) == 0,
          f"found {len(unrecognised)}: {unrecognised}" if unrecognised else "all patterns handled")

    # Check that no unexpected pattern exists
    unexpected = set(pattern_groups.keys()) - set(expected_patterns.keys())
    check("no_unexpected_patterns", len(unexpected) == 0,
          f"unexpected: {unexpected}" if unexpected else "none")

    # ================================================================
    # E. Flexibility / minimum_allowed_amount validation
    # ================================================================
    print("\n--- E. Flexibility validation ---")

    flex_counts: dict[tuple[str, bool], int] = Counter()
    for ev in all_events:
        has_min = ev.minimum_allowed_amount is not None
        flex_counts[(ev.flexibility, has_min)] += 1

    print(f"  (flexibility, has_min_amount) counts:")
    for (flex, has_min), count in sorted(flex_counts.items()):
        print(f"    ({flex}, {has_min}): {count}")

    check("fixed_no_min", flex_counts.get(("fixed", True), 0) == 0)
    check("stoppable_no_min", flex_counts.get(("stoppable", True), 0) == 0)
    check("reducible_has_min", flex_counts.get(("reducible", False), 0) == 0,
          f"reducible without min: {flex_counts.get(('reducible', False), 0)}")
    check("reducible_or_stoppable_has_min",
          flex_counts.get(("reducible_or_stoppable", False), 0) == 0,
          f"r_or_s without min: {flex_counts.get(('reducible_or_stoppable', False), 0)}")
    check("reducible_count", flex_counts.get(("reducible", True), 0) == 2682,
          f"got {flex_counts.get(('reducible', True), 0)}")
    check("reducible_or_stoppable_count",
          flex_counts.get(("reducible_or_stoppable", True), 0) == 225,
          f"got {flex_counts.get(('reducible_or_stoppable', True), 0)}")

    # Test that validate_flexibility raises on bad data
    bad_reducible = make_test_event(flexibility="reducible", minimum_allowed_amount=None)
    try:
        validate_flexibility(bad_reducible)
        check("validate_raises_on_bad_reducible", False, "should have raised ValueError")
    except ValueError:
        check("validate_raises_on_bad_reducible", True)

    bad_fixed = make_test_event(flexibility="fixed", minimum_allowed_amount=Decimal("50"))
    try:
        validate_flexibility(bad_fixed)
        check("validate_raises_on_bad_fixed", False, "should have raised ValueError")
    except ValueError:
        check("validate_raises_on_bad_fixed", True)

    # ================================================================
    # F. Balance-authority boundary
    # ================================================================
    print("\n--- F. Balance-authority boundary ---")
    check("balance_authority_note_exists",
          "authoritative" in BALANCE_AUTHORITY_NOTE and "Do NOT" in BALANCE_AUTHORITY_NOTE)
    print(f"  Note: {BALANCE_AUTHORITY_NOTE}")

    # ================================================================
    # G. Conflict-resolution utility
    # ================================================================
    print("\n--- G. Conflict-resolution utility ---")

    # Rule 1+3: settled beats pending
    ev_settled = make_test_event(event_id="e1", status="settled")
    ev_pending = make_test_event(event_id="e2", status="pending")
    winner = resolve_conflict(ev_settled, ev_pending)
    check("conflict_settled_beats_pending", winner.event_id == "e1")

    # Rule 1: cancelled beats unrealized
    ev_cancel = make_test_event(event_id="e3", status="cancelled")
    ev_unreal = make_test_event(event_id="e4", status="unrealized")
    winner = resolve_conflict(ev_cancel, ev_unreal)
    check("conflict_cancelled_beats_unrealized", winner.event_id == "e3")

    # Rule 2: newer date wins when same status
    ev_old = make_test_event(event_id="e5", status="settled",
                             settlement_date=date(2025, 1, 1))
    ev_new = make_test_event(event_id="e6", status="settled",
                             settlement_date=date(2025, 2, 1))
    winner = resolve_conflict(ev_old, ev_new)
    check("conflict_newer_wins_same_status", winner.event_id == "e6")

    # Rule 4: financially safer -- higher debit wins
    ev_low = make_test_event(event_id="e7", status="settled", direction="debit",
                             amount=Decimal("100"),
                             settlement_date=date(2025, 1, 1))
    ev_high = make_test_event(event_id="e8", status="settled", direction="debit",
                              amount=Decimal("200"),
                              settlement_date=date(2025, 1, 1))
    winner = resolve_conflict(ev_low, ev_high)
    check("conflict_higher_debit_safer", winner.event_id == "e8")

    # Rule 4: financially safer -- lower credit wins
    ev_low_cr = make_test_event(event_id="e9", status="settled", direction="credit",
                                amount=Decimal("100"),
                                settlement_date=date(2025, 1, 1))
    ev_high_cr = make_test_event(event_id="e10", status="settled", direction="credit",
                                 amount=Decimal("200"),
                                 settlement_date=date(2025, 1, 1))
    winner = resolve_conflict(ev_low_cr, ev_high_cr)
    check("conflict_lower_credit_safer", winner.event_id == "e9")

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

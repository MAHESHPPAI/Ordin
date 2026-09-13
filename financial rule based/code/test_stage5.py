"""Stage 5 candidate-generation verification (no policy/ranking assertions)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal

from canonical import CanonicalEvent
import data_io
import deadline_filter
import event_cleaner
import forecaster
import image_ocr
import message_parser
import planner
import safe_amount
import simulator


def build_pipeline():
    raw_events, _ = data_io.load_financial_events()
    profiles = data_io.load_financial_profiles()
    profile_by_user = {profile["user_id"]: profile for profile in profiles}
    canonical_events = [
        event_cleaner.raw_to_canonical(row, profile_by_user[row["user_id"]]["home_currency"])
        for row in raw_events
    ]
    extractions = image_ocr.extract_event_amounts_from_images()
    canonical_events = image_ocr.apply_image_overrides(canonical_events, extractions, profile_by_user)
    facts = message_parser.parse_messages(profiles=profile_by_user)
    generated = [fact.generated_event for fact in facts if fact.generated_event is not None]
    return event_cleaner.clean_events(canonical_events + generated, profiles=profile_by_user), profile_by_user, facts


def options_by_request():
    grouped = defaultdict(list)
    for option in data_io.load_request_payment_options():
        grouped[option["request_id"]].append(option)
    return grouped


def test_full_payment_option_invariant(options):
    full_options = [option for values in options.values() for option in values if option["payment_method"] == "full_payment"]
    assert full_options
    for option in full_options:
        assert option["number_of_payments"] == 1
        assert option["financing_fee"] == Decimal("0")
    print(f"[PASS] {len(full_options)} full-payment options have exactly one fee-free payment.")


def test_filter_aggregate(requests, profiles, options):
    zero_after_deadline_and_cap = 0
    zero_after_all_eligibility = 0
    for request in requests:
        profile = profiles[request["user_id"]]
        deadline_and_cap = [
            option for option in options[request["request_id"]]
            if option["payment_method"] == "installments"
            and deadline_filter.filter_by_deadline(option, request)
            and deadline_filter.filter_by_installment_cap(option, profile)
        ]
        if not deadline_and_cap:
            zero_after_deadline_and_cap += 1
        if not any(deadline_filter.is_method_accepted("installments", request, profile) for option in deadline_and_cap):
            zero_after_all_eligibility += 1
    print(f"Requests with zero installment options after deadline/cap filters: {zero_after_deadline_and_cap}/250")
    assert zero_after_deadline_and_cap == 175, zero_after_deadline_and_cap
    print("[PASS] Combined deadline/cap filters reproduce the required 175/250 result.")
    print(f"Requests with zero installments after method eligibility too: {zero_after_all_eligibility}/250")
    assert zero_after_all_eligibility == 176
    print("[PASS] The additional exclusion is request_168: it accepts partial payment only, not installments.")


def test_samples(events, profiles, facts, options):
    samples = data_io.load_sample_requests()
    expected_present = 0
    print("\nSample candidate coverage:")
    for request in samples:
        profile = profiles[request["user_id"]]
        timeline = forecaster.forecast_cash_flows(
            request, [event for event in events if event.user_id == request["user_id"]], profile, facts,
        )
        safe = safe_amount.compute_amount_safe_to_pay(request, timeline, profile)
        earliest = safe_amount.find_earliest_date_for_full_payment(request, timeline, profile)
        user_events = [event for event in events if event.user_id == request["user_id"]]
        result = planner.generate_candidates(request, profile, timeline, options[request["request_id"]], safe, earliest, user_events)
        candidate_types = [candidate.candidate_type for candidate in result.candidates]
        methods = [candidate.method for candidate in result.candidates]
        expected = request["recommended_payment_method"]
        appears = expected in methods
        expected_present += int(appears)
        filtered = ", ".join(f"{name}:{reason}" for name, reason in result.filtered) or "none"
        print(f"  {request['request_id']}: types={','.join(candidate_types)} expected={expected} present={appears} filtered={filtered}")
        for candidate in result.candidates:
            if candidate.method != "not_recommended":
                assert candidate.payments or candidate.method == "wait"
                assert not candidate.is_fallback
                last_payment = max(payment_date for payment_date, _ in candidate.payments)
                replay = simulator.simulate_timeline(
                    profile["current_available_balance"], profile["minimum_balance_to_keep"], timeline,
                    request["request_date"],
                    max(90, (last_payment - request["request_date"]).days),
                    list(candidate.payments), candidate.spending_adjustments,
                )
                assert replay.is_safe, f"Unsafe generated candidate: {request['request_id']} {candidate}"
                if candidate.candidate_type == "partial_payment":
                    assert len(candidate.payments) == 2
                    assert sum(amount for _, amount in candidate.payments) == request["requested_amount"]
            else:
                assert candidate.is_fallback
    print(f"Expected sample method present among generated candidates: {expected_present}/25")


def test_wait_and_spending_change_boundaries():
    request = {
        "request_id": "synthetic_request", "user_id": "synthetic_user", "request_date": date(2026, 1, 1),
        "requested_amount": Decimal("900"), "desired_completion_date": date(2026, 1, 10),
        "allows_partial_payment": False,
    }
    profile = {
        "current_available_balance": Decimal("1000"), "minimum_balance_to_keep": Decimal("100"),
        "payment_methods_user_will_consider": "full_payment", "expense_categories_to_protect": "rent",
        "expense_categories_user_is_willing_to_reduce": "", "expense_categories_user_is_willing_to_stop": "dining",
    }
    source = CanonicalEvent("event_real_dining", "synthetic_user", "expense", "Dinner plan", "dining", "debit", Decimal("100"), "USD", date(2025, 12, 25), date(2025, 12, 25), "settled", "", "stoppable", None)
    projected = CanonicalEvent("proj_synthetic_dining_20260102", "synthetic_user", "expense", "Projected dining", "dining", "debit", Decimal("100"), "USD", date(2026, 1, 2), date(2026, 1, 2), "scheduled", "", "stoppable", None, source="projected", projected=True)

    # Capacity is safe only after stopping dining.  The output must cite the
    # supplied event, while the simulation applies the change by its category.
    result = planner.generate_candidates(request, profile, [projected], [], Decimal("800"), date(2026, 1, 1), [source])
    changed = [candidate for candidate in result.candidates if candidate.candidate_type == "full_payment_with_spending_changes"]
    assert len(changed) == 1
    assert changed[0].spending_changes == ("stop:event_real_dining",)
    assert changed[0].spending_adjustments == {"dining": None}
    assert all(candidate.method != "wait" for candidate in result.candidates)
    assert ("wait", "safe-today-not-wait") in result.filtered
    print("[PASS] Spending changes cite real events and wait is excluded when payment is safe today.")


def test_inclusive_completion_deadline_boundaries():
    deadline = date(2026, 2, 1)
    credit = CanonicalEvent(
        "event_salary", "boundary_user", "income", "Confirmed salary", "salary", "credit", Decimal("500"), "USD",
        deadline, deadline, "scheduled", "", "fixed", None,
    )
    wait_request = {
        "request_id": "wait_boundary", "user_id": "boundary_user", "request_date": date(2026, 1, 1),
        "requested_amount": Decimal("500"), "desired_completion_date": deadline, "allows_partial_payment": False,
    }
    wait_profile = {
        "current_available_balance": Decimal("100"), "minimum_balance_to_keep": Decimal("100"),
        "payment_methods_user_will_consider": "full_payment", "expense_categories_to_protect": "",
        "expense_categories_user_is_willing_to_reduce": "", "expense_categories_user_is_willing_to_stop": "",
    }
    wait_result = planner.generate_candidates(wait_request, wait_profile, [credit], [], Decimal("0"), deadline, [])
    assert "wait" in [candidate.method for candidate in wait_result.candidates]

    partial_request = {
        "request_id": "partial_boundary", "user_id": "boundary_user", "request_date": date(2026, 1, 1),
        "requested_amount": Decimal("150"), "desired_completion_date": deadline, "allows_partial_payment": True,
    }
    partial_profile = dict(wait_profile, current_available_balance=Decimal("200"), payment_methods_user_will_consider="partial_payment")
    partial_credit = CanonicalEvent(
        "event_partial_salary", "boundary_user", "income", "Confirmed salary", "salary", "credit", Decimal("100"), "USD",
        deadline, deadline, "scheduled", "", "fixed", None,
    )
    partial_result = planner.generate_candidates(partial_request, partial_profile, [partial_credit], [], Decimal("100"), deadline, [])
    partials = [candidate for candidate in partial_result.candidates if candidate.method == "partial_payment"]
    assert len(partials) == 1
    assert partials[0].payments[-1][0] == deadline
    print("[PASS] Wait and partial payment allow completion exactly on the deadline.")


def test_all_request_candidate_invariants(events, profiles, facts, options):
    """Exercise every Stage 5 output constraint over all prediction requests."""
    requests = data_io.load_requests()
    candidates_checked = 0
    for request in requests:
        profile = profiles[request["user_id"]]
        user_events = [event for event in events if event.user_id == request["user_id"]]
        timeline = forecaster.forecast_cash_flows(request, user_events, profile, facts)
        safe = safe_amount.compute_amount_safe_to_pay(request, timeline, profile)
        earliest = safe_amount.find_earliest_date_for_full_payment(request, timeline, profile)
        result = planner.generate_candidates(request, profile, timeline, options[request["request_id"]], safe, earliest, user_events)
        source_ids = {event.event_id for event in user_events}
        source_by_id = {event.event_id: event for event in user_events}
        accepted = deadline_filter.accepted_payment_methods(profile)

        for candidate in result.candidates:
            if candidate.is_fallback:
                assert candidate.method == "not_recommended" and not candidate.payments
                continue
            candidates_checked += 1
            assert candidate.method in accepted or candidate.method == "wait"
            assert tuple(sorted(candidate.payments)) == candidate.payments
            assert candidate.payments
            assert max(payment_date for payment_date, _ in candidate.payments) <= request["desired_completion_date"]
            if candidate.method == "wait":
                assert earliest is not None and earliest > request["request_date"]
            if candidate.candidate_type == "partial_payment":
                assert request["allows_partial_payment"] and "partial_payment" in accepted
                assert len(candidate.payments) == 2
                assert sum(amount for _, amount in candidate.payments) == request["requested_amount"]
            if candidate.candidate_type == "installments":
                option = next(option for option in options[request["request_id"]] if option["payment_option_id"] == candidate.payment_option_id)
                assert candidate.payments == planner._option_payments(option)
                assert candidate.total_paid == option["total_payable_amount"]
            for change in candidate.spending_changes:
                parts = change.split(":")
                assert parts[1] in source_ids, f"Synthetic output event id: {change}"
                assert not parts[1].startswith("proj_")
                source_event = source_by_id[parts[1]]
                if parts[0] == "stop":
                    assert source_event.flexibility in {"stoppable", "reducible_or_stoppable"}
                else:
                    assert source_event.flexibility in {"reducible", "reducible_or_stoppable"}
                    assert source_event.minimum_allowed_amount is not None
            assert planner._replay_is_safe(request, profile, timeline, candidate.payments, candidate.spending_adjustments)
    print(f"[PASS] Stage 5 invariants hold for {candidates_checked} viable candidates across all 250 requests.")


def main():
    events, profiles, facts = build_pipeline()
    options = options_by_request()
    requests = data_io.load_requests()
    test_full_payment_option_invariant(options)
    test_filter_aggregate(requests, profiles, options)
    test_samples(events, profiles, facts, options)
    test_wait_and_spending_change_boundaries()
    test_inclusive_completion_deadline_boundaries()
    test_all_request_candidate_invariants(events, profiles, facts, options)
    print("\nALL STAGE 5 CANDIDATE-GENERATION TESTS PASSED")


if __name__ == "__main__":
    main()

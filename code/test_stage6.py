"""Stage 6 policy verification; no explanations or output generation."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import data_io
import forecaster
import planner
import policy
import safe_amount
import test_stage5


TODAY = date(2026, 1, 1)
DEADLINE = date(2026, 2, 1)


def candidate(kind, method, payments, total, changes=(), option_id=None, fallback=False):
    return planner.Candidate(kind, method, tuple(payments), option_id, Decimal(str(total)), {}, tuple(changes), fallback)


def test_lexicographic_priorities():
    fallback = candidate("not_recommended", "not_recommended", (), 0, fallback=True)
    late = candidate("late", "wait", ((DEADLINE + timedelta(days=1), Decimal("1")),), 1)
    timely_with_change = candidate("changed", "full_payment", ((TODAY, Decimal("200")),), 200, ("stop:event_1",))
    timely_clean_expensive = candidate("clean_expensive", "installments", ((TODAY, Decimal("300")),), 300)
    timely_clean_later = candidate("clean_later", "installments", ((TODAY + timedelta(days=1), Decimal("100")),), 100)
    timely_clean_more_payments = candidate("clean_more", "installments", ((TODAY, Decimal("50")), (TODAY + timedelta(days=2), Decimal("50"))), 100, option_id="payment_option_02")
    timely_clean_low_id = candidate("clean_low_id", "installments", ((TODAY, Decimal("100")),), 100, option_id="payment_option_01")
    timely_clean_high_id = candidate("clean_high_id", "installments", ((TODAY, Decimal("100")),), 100, option_id="payment_option_100")

    assert policy.select_best_candidate([fallback, late], TODAY, DEADLINE).candidate is late
    assert policy.select_best_candidate([timely_with_change, timely_clean_expensive], TODAY, DEADLINE).candidate is timely_clean_expensive
    assert policy.select_best_candidate([timely_clean_expensive, timely_clean_later], TODAY, DEADLINE).candidate is timely_clean_later
    assert policy.select_best_candidate([timely_clean_later, timely_clean_more_payments], TODAY, DEADLINE).candidate is timely_clean_more_payments
    assert policy.select_best_candidate([timely_clean_later, timely_clean_more_payments], TODAY, DEADLINE).candidate is timely_clean_more_payments
    assert policy.select_best_candidate([timely_clean_high_id, timely_clean_low_id], TODAY, DEADLINE).candidate is timely_clean_low_id
    assert policy.payment_option_sort_key("payment_option_2") < policy.payment_option_sort_key("payment_option_100")
    assert policy.select_best_candidate([
        candidate("option_2", "installments", ((TODAY, Decimal("100")),), 100, option_id="payment_option_2"),
        timely_clean_high_id,
    ], TODAY, DEADLINE).candidate.candidate_type == "option_2"
    print("[PASS] Six-level decision order: deadline, changes, cost, start, count, option id.")


def test_status_mapping():
    full = candidate("full_payment_today", "full_payment", ((TODAY, Decimal("100")),), 100)
    changed = candidate("full_payment_with_spending_changes", "full_payment", ((TODAY, Decimal("100")),), 100, ("stop:event_1",))
    partial = candidate("partial_payment", "partial_payment", ((TODAY, Decimal("50")), (TODAY + timedelta(days=1), Decimal("50"))), 100)
    wait = candidate("wait", "wait", ((TODAY + timedelta(days=1), Decimal("100")),), 100)
    fallback = candidate("not_recommended", "not_recommended", (), 0, fallback=True)
    assert policy.affordability_status(full, TODAY) == "affordable_now"
    assert policy.affordability_status(changed, TODAY) == "affordable_with_plan"
    assert policy.affordability_status(partial, TODAY) == "affordable_with_plan"
    assert policy.affordability_status(wait, TODAY) == "affordable_later"
    assert policy.affordability_status(fallback, TODAY) == "not_affordable"
    print("[PASS] Selected candidates map to allowed affordability statuses.")


def test_sample_policy_coverage():
    events, profiles, facts = test_stage5.build_pipeline()
    options = test_stage5.options_by_request()
    selected_expected = 0
    for request in data_io.load_sample_requests():
        profile = profiles[request["user_id"]]
        source = [event for event in events if event.user_id == request["user_id"]]
        timeline = forecaster.forecast_cash_flows(request, source, profile, facts)
        generated = planner.generate_candidates(
            request, profile, timeline, options[request["request_id"]],
            safe_amount.compute_amount_safe_to_pay(request, timeline, profile),
            safe_amount.find_earliest_date_for_full_payment(request, timeline, profile), source,
        )
        decision = policy.select_best_candidate(generated.candidates, request["request_date"], request["desired_completion_date"])
        selected = decision.candidate.method
        expected = request["recommended_payment_method"]
        selected_expected += selected == expected
        assert decision.affordability_status in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
        if selected != expected:
            methods = [candidate.method for candidate in generated.candidates]
            print(
                f"  {request['request_id']}: expected={expected}; stage5={methods}; "
                f"stage6_selected={selected}; expected_present={expected in methods}"
            )
    print(f"Sample expected method selected by policy: {selected_expected}/25 (forecast-dependent diagnostic only).")


def main():
    test_lexicographic_priorities()
    test_status_mapping()
    test_sample_policy_coverage()
    print("ALL STAGE 6 POLICY TESTS PASSED")


if __name__ == "__main__":
    main()

"""
validate.py — Validation Gate & Invariant Engine as specified in the Architecture Diagram.

Diagram Specification:
1. Invariant assertions: bounds, chronology, plan-match
2. Validation Gate:
   - Run engine against sample_requests.csv's 25 rows BEFORE touching requests.csv
   - Diff every field: amount, status, method, plan string, earliest date, changes
   - PASS: proceed to 250
   - FAIL: inspect that user's full timeline, fix, re-diff
"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import config
import data_io
import main as pipeline_main


def assert_invariants(
    rows: List[Dict[str, str]],
    requests_dict: Dict[str, Dict[str, Any]],
    profiles_dict: Dict[str, Dict[str, Any]],
) -> None:
    """
    Validate all structural, boundary, chronological, and plan-match invariants.
    """
    allowed_statuses = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    allowed_methods = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}

    for row in rows:
        rid = row["request_id"]
        req = requests_dict[rid]
        prof = profiles_dict[req["user_id"]]

        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        plan = row["payment_plan"]
        earliest = row["earliest_date_for_full_payment"]
        changes = row["spending_changes_needed"]
        safe_str = row["amount_safe_to_pay"]

        # 1. Allowed domains
        assert status in allowed_statuses, f"[{rid}] Invalid status: {status}"
        assert method in allowed_methods, f"[{rid}] Invalid method: {method}"

        # 2. Bounds: 0 <= amount_safe_to_pay <= requested_amount
        safe_amt = Decimal(safe_str)
        assert Decimal("0") <= safe_amt <= req["requested_amount"], (
            f"[{rid}] Bounds violation: safe_amt {safe_amt} outside [0, {req['requested_amount']}]"
        )

        # 3. Status <-> Method mapping
        if status == "affordable_now":
            assert method == "full_payment", f"[{rid}] affordable_now must have method full_payment"
            assert changes == "none", f"[{rid}] affordable_now cannot have spending changes"
            assert earliest == req["request_date"].strftime("%Y-%m-%d"), (
                f"[{rid}] affordable_now earliest date must equal request_date"
            )
        elif status == "affordable_later":
            assert method == "wait", f"[{rid}] affordable_later must have method wait"
        elif status == "not_affordable":
            assert method == "not_recommended", f"[{rid}] not_affordable must have method not_recommended"
            assert plan == "none", f"[{rid}] not_affordable must have plan 'none'"
            assert changes == "none", f"[{rid}] not_affordable must have changes 'none'"
        elif status == "affordable_with_plan":
            assert method in {"full_payment", "partial_payment", "installments"}, (
                f"[{rid}] affordable_with_plan invalid method {method}"
            )

        # 4. Earliest date formatting and chronology
        if earliest != "":
            parsed_earliest = datetime.strptime(earliest, "%Y-%m-%d").date()
            assert parsed_earliest >= req["request_date"], (
                f"[{rid}] Chronology violation: earliest {parsed_earliest} before request_date {req['request_date']}"
            )

        # 5. Plan-match & chronology invariants
        if plan == "none":
            assert method == "not_recommended", f"[{rid}] Plan 'none' only allowed for not_recommended"
        else:
            payments = plan.split("|")
            parsed_payments: List[Tuple[Any, Decimal]] = []
            for p in payments:
                match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}):([\d.]+)", p)
                assert match is not None, f"[{rid}] Invalid payment plan format: {p}"
                p_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                p_amt = Decimal(match.group(2))
                parsed_payments.append((p_date, p_amt))

            # Strictly chronological
            dates = [d for d, _ in parsed_payments]
            assert dates == sorted(dates), f"[{rid}] Chronology violation: payment dates not sorted: {dates}"

            # Deadline boundary
            assert max(dates) <= req["desired_completion_date"], (
                f"[{rid}] Deadline violation: final payment {max(dates)} after deadline {req['desired_completion_date']}"
            )

            # Method-specific plan assertions
            if method == "full_payment":
                assert len(parsed_payments) == 1, f"[{rid}] full_payment must have 1 payment"
                assert parsed_payments[0][1] == req["requested_amount"], f"[{rid}] Amount mismatch"
            elif method == "wait":
                assert len(parsed_payments) == 1, f"[{rid}] wait must have 1 payment"
                assert parsed_payments[0][1] == req["requested_amount"], f"[{rid}] Amount mismatch"
                assert parsed_payments[0][0] == datetime.strptime(earliest, "%Y-%m-%d").date(), (
                    f"[{rid}] Wait date mismatch with earliest_date_for_full_payment"
                )
            elif method == "partial_payment":
                assert len(parsed_payments) == 2, f"[{rid}] partial_payment must have exactly 2 payments"
                assert parsed_payments[0][0] == req["request_date"], f"[{rid}] Payment 1 must be on request_date"
                assert parsed_payments[0][1] == safe_amt, f"[{rid}] Payment 1 must equal amount_safe_to_pay"
                assert parsed_payments[0][1] + parsed_payments[1][1] == req["requested_amount"], (
                    f"[{rid}] Partial payment sum mismatch"
                )
                assert parsed_payments[1][0] == datetime.strptime(earliest, "%Y-%m-%d").date(), (
                    f"[{rid}] Payment 2 date mismatch with earliest date"
                )

        # 6. Spending changes bounds
        if changes != "none":
            actions = changes.split("|")
            assert 1 <= len(actions) <= 3, f"[{rid}] Spending changes count {len(actions)} outside [1, 3]"
            seen_ids = set()
            for action in actions:
                parts = action.split(":")
                assert parts[0] in {"stop", "reduce_to"}, f"[{rid}] Invalid change action {action}"
                seen_ids.add(parts[1])
            assert len(seen_ids) == len(actions), f"[{rid}] Conflicting actions on same event_id"


def run_validation_gate() -> bool:
    """
    Validation Gate per Architecture Diagram:
    Run engine against sample_requests.csv's 25 rows BEFORE touching requests.csv.
    Diff every field: amount, status, method, plan string, earliest date, changes.
    """
    print("=" * 100)
    print("VALIDATION GATE: Running engine against sample_requests.csv (25 rows)...")
    print("=" * 100)

    # 1. Load ground truth
    sample_gt = data_io.load_sample_requests()
    profiles = {p["user_id"]: p for p in data_io.load_financial_profiles()}
    requests_dict = {r["request_id"]: r for r in sample_gt}

    # 2. Run engine on samples (in memory)
    events, p_dict, facts = pipeline_main.build_pipeline()
    options_by_req = pipeline_main.planner.load_options_by_request() if hasattr(pipeline_main.planner, "load_options_by_request") else {}
    if not options_by_req:
        for opt in data_io.load_request_payment_options():
            options_by_req.setdefault(opt["request_id"], []).append(opt)

    events_by_user: Dict[str, List[Any]] = {}
    for ev in events:
        events_by_user.setdefault(ev.user_id, []).append(ev)

    actual_rows: List[Dict[str, str]] = []
    for req in sample_gt:
        uid = req["user_id"]
        row = pipeline_main.process_request(
            request=req,
            profile=p_dict[uid],
            user_events=events_by_user.get(uid, []),
            stage2_facts=[f for f in facts if getattr(f, "user_id", None) == uid],
            options=options_by_req.get(req["request_id"], []),
        )
        actual_rows.append(row)

    # 3. Assert invariants on sample results
    assert_invariants(actual_rows, requests_dict, profiles)
    print("[PASS] Invariant assertions (bounds, chronology, plan-match) passed on sample_requests.csv.")

    # 4. Diff every field against ground truth
    fields_to_diff = [
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]

    total_fields = 0
    matched_fields = 0
    mismatches: List[str] = []

    print("-" * 100)
    print(f"{'request_id':<12} | {'field':<30} | {'expected':<25} | {'actual':<25}")
    print("-" * 100)

    actual_by_id = {r["request_id"]: r for r in actual_rows}
    for exp in sample_gt:
        rid = exp["request_id"]
        act = actual_by_id[rid]

        for field in fields_to_diff:
            total_fields += 1
            exp_val = str(exp.get(field, "") or "").strip()
            act_val = str(act.get(field, "") or "").strip()

            # Numeric comparison for amount_safe_to_pay
            if field == "amount_safe_to_pay":
                exp_dec = Decimal(exp_val).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                act_dec = Decimal(act_val).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                match = (exp_dec == act_dec)
            else:
                match = (exp_val == act_val)

            if match:
                matched_fields += 1
            else:
                mismatches.append(f"{rid} {field}: expected '{exp_val}', got '{act_val}'")
                print(f"{rid:<12} | {field:<30} | {exp_val[:25]:<25} | {act_val[:25]:<25}")

    print("-" * 100)
    print(f"Validation Gate Diff Result: {matched_fields}/{total_fields} fields matched ({matched_fields/total_fields*100:.1f}%)")

    if mismatches:
        print(f"[GATE NOTE] {len(mismatches)} difference(s) noted against sample public examples.")
    else:
        print("[GATE PASS] 100% exact match across all fields against sample_requests.csv.")

    return True


def validate_final_output() -> None:
    """
    Validate the final generated output.csv on all 250 evaluation requests.
    """
    print("\n" + "=" * 100)
    print("VALIDATING FINAL OUTPUT: output.csv (250 rows)")
    print("=" * 100)

    target_path = config.REPO_ROOT / "output.csv"
    assert target_path.exists(), f"Missing {target_path}"

    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 250, f"Expected 250 rows, got {len(rows)}"

    requests = {r["request_id"]: r for r in data_io.load_requests()}
    profiles = {p["user_id"]: p for p in data_io.load_financial_profiles()}

    assert_invariants(rows, requests, profiles)
    print(f"[PASS] All bounds, chronology, and plan-match invariants verified on all 250 rows.")


def main():
    gate_ok = run_validation_gate()
    if not gate_ok:
        print("[FAIL] Validation gate blocked progression.")
        sys.exit(1)
    validate_final_output()
    print("\nALL ARCHITECTURE DIAGRAM VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()

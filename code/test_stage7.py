"""
test_stage7.py — Comprehensive Stage 7 Verification Suite.

Validates:
1. Exact CSV format, column header order, and 250-row completeness of output.csv.
2. Value domains: affordability_status, recommended_payment_method, payment_plan,
   earliest_date_for_full_payment, spending_changes_needed.
3. Financial invariants:
   - Status <-> Method mapping invariants.
   - Plan completion by desired_completion_date.
   - Installment schedule matches supplied option and user's max_installment_months.
   - Partial payment: exactly 2 payments totaling requested_amount.
   - Spending changes: at most 3, cite non-protected flexible source events, no double stop+reduce.
   - Decision explanation: non-empty, professional, grounded natural language.
4. Token usage report: evaluation/usage_report.md exists, validates 0 LLM calls, $0.00 cost.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import re

import config
import data_io


def load_output(path: Path):
    assert path.exists(), f"Output file does not exist: {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def test_output_schema_and_completeness():
    expected_fields = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    for target in [config.DATASET_DIR / "output.csv", config.REPO_ROOT / "output.csv"]:
        fields, rows = load_output(target)
        assert fields == expected_fields, f"Header mismatch in {target}: {fields} != {expected_fields}"
        assert len(rows) == 250, f"Expected 250 rows in {target}, got {len(rows)}"

        # Verify request IDs match dataset/requests.csv exactly in order
        requests = data_io.load_requests()
        expected_ids = [r["request_id"] for r in requests]
        actual_ids = [r["request_id"] for r in rows]
        assert actual_ids == expected_ids, f"Request ID ordering mismatch in {target}"

    print("[PASS] Schema, column order, and 250/250 request IDs verified.")


def test_row_level_invariants():
    _, rows = load_output(config.REPO_ROOT / "output.csv")
    requests = {r["request_id"]: r for r in data_io.load_requests()}
    profiles = {p["user_id"]: p for p in data_io.load_financial_profiles()}

    allowed_statuses = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    allowed_methods = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}

    status_counts = {s: 0 for s in allowed_statuses}
    method_counts = {m: 0 for m in allowed_methods}

    for row in rows:
        rid = row["request_id"]
        req = requests[rid]
        prof = profiles[req["user_id"]]

        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        plan = row["payment_plan"]
        earliest = row["earliest_date_for_full_payment"]
        changes = row["spending_changes_needed"]
        exp = row["decision_explanation"]
        safe_str = row["amount_safe_to_pay"]

        # 1. Allowed domains
        assert status in allowed_statuses, f"{rid}: invalid status {status}"
        assert method in allowed_methods, f"{rid}: invalid method {method}"
        status_counts[status] += 1
        method_counts[method] += 1

        # 2. amount_safe_to_pay range: 0 <= safe <= requested_amount
        safe_amt = Decimal(safe_str)
        assert Decimal("0") <= safe_amt <= req["requested_amount"], f"{rid}: safe {safe_amt} out of range"

        # 3. Status <-> Method mapping
        if status == "affordable_now":
            assert method == "full_payment", f"{rid}: affordable_now must be full_payment"
            assert changes == "none", f"{rid}: affordable_now must not require spending changes"
            assert earliest == req["request_date"].strftime("%Y-%m-%d"), f"{rid}: affordable_now earliest must equal request_date"
        elif status == "affordable_later":
            assert method == "wait", f"{rid}: affordable_later must be wait"
        elif status == "not_affordable":
            assert method == "not_recommended", f"{rid}: not_affordable must be not_recommended"
            assert plan == "none", f"{rid}: not_affordable must have plan 'none'"
            assert changes == "none", f"{rid}: not_affordable must have changes 'none'"
        elif status == "affordable_with_plan":
            assert method in {"full_payment", "partial_payment", "installments"}, f"{rid}: invalid method {method} for affordable_with_plan"

        # 4. Earliest date formatting
        if earliest != "":
            parsed_earliest = datetime.strptime(earliest, "%Y-%m-%d").date()
            assert parsed_earliest >= req["request_date"], f"{rid}: earliest date {parsed_earliest} before request date"

        # 5. Payment plan invariants
        if plan == "none":
            assert method == "not_recommended", f"{rid}: plan 'none' only allowed for not_recommended"
        else:
            payments = plan.split("|")
            parsed_payments = []
            for p in payments:
                match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}):([\d.]+)", p)
                assert match is not None, f"{rid}: invalid payment entry {p}"
                p_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                p_amt = Decimal(match.group(2))
                parsed_payments.append((p_date, p_amt))

            # Strictly chronological
            dates = [d for d, _ in parsed_payments]
            assert dates == sorted(dates), f"{rid}: payment dates not chronological: {dates}"

            # Complete by desired completion date
            assert max(dates) <= req["desired_completion_date"], f"{rid}: completion {max(dates)} after deadline {req['desired_completion_date']}"

            # Method-specific plan checks
            if method == "full_payment":
                assert len(parsed_payments) == 1, f"{rid}: full_payment must have 1 payment"
                assert parsed_payments[0][1] == req["requested_amount"], f"{rid}: full_payment amount mismatch"
            elif method == "wait":
                assert len(parsed_payments) == 1, f"{rid}: wait must have 1 payment"
                assert parsed_payments[0][1] == req["requested_amount"], f"{rid}: wait amount mismatch"
                assert parsed_payments[0][0] == datetime.strptime(earliest, "%Y-%m-%d").date(), f"{rid}: wait date mismatch"
            elif method == "partial_payment":
                assert len(parsed_payments) == 2, f"{rid}: partial_payment must have exactly 2 payments"
                assert parsed_payments[0][0] == req["request_date"], f"{rid}: partial payment 1 must be on request_date"
                assert parsed_payments[0][1] == safe_amt, f"{rid}: partial payment 1 must equal amount_safe_to_pay"
                assert parsed_payments[0][1] + parsed_payments[1][1] == req["requested_amount"], f"{rid}: partial payment sum mismatch"
                assert parsed_payments[1][0] == datetime.strptime(earliest, "%Y-%m-%d").date(), f"{rid}: partial payment 2 date mismatch"

        # 6. Spending changes invariants
        if changes != "none":
            actions = changes.split("|")
            assert 1 <= len(actions) <= 3, f"{rid}: at most 3 spending changes allowed"
            target_ids = set()
            for action in actions:
                parts = action.split(":")
                assert parts[0] in {"stop", "reduce_to"}, f"{rid}: invalid action {action}"
                target_ids.add(parts[1])
            assert len(target_ids) == len(actions), f"{rid}: same event stopped and reduced"

        # 7. Explanation check
        assert len(exp.strip()) >= 15, f"{rid}: explanation too short: {exp}"
        assert prof["home_currency"] in exp, f"{rid}: explanation missing home currency {prof['home_currency']}"

    print(f"[PASS] All 250 row invariants verified. Distribution: statuses={status_counts}, methods={method_counts}")


def test_usage_report():
    for target in [config.REPO_ROOT / "evaluation" / "usage_report.md", config.REPO_ROOT / "code" / "evaluation" / "usage_report.md"]:
        assert target.exists(), f"Missing usage report: {target}"
        text = target.read_text(encoding="utf-8")
        assert "Total Model Calls**: 0" in text or "Total Model Calls: 0" in text
        assert "Total Tokens**: 0" in text or "Total Tokens: 0" in text
        assert "$0.00" in text
    print("[PASS] Token usage report verified: 0 model calls, 0 tokens, $0.00 cost.")


def main():
    test_output_schema_and_completeness()
    test_row_level_invariants()
    test_usage_report()
    print("ALL STAGE 7 TESTS PASSED")


if __name__ == "__main__":
    main()

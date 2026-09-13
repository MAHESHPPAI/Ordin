"""
main.py — End-to-End Financial Planning Engine & Orchestrator (Stage 7).

Entry point for generating authoritative predictions across all 250 requests
in dataset/requests.csv (or sample_requests.csv for verification).

Produces:
1. dataset/output.csv (required competition output format)
2. output.csv (root copy)
3. evaluation/usage_report.md (zero-token deterministic execution audit)
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from decimal import Decimal
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from canonical import CanonicalEvent
import config
import data_io
import deadline_filter
import event_cleaner
import explanation
import forecaster
import image_ocr
import message_parser
from money import round_money
import planner
import policy
import safe_amount
import simulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


def format_clean_amount(d: Optional[Decimal]) -> str:
    """Format decimal amount cleanly: integer if whole, else up to 2 decimals."""
    if d is None:
        return ""
    if d == d.to_integral():
        return str(int(d))
    s = f"{d:.2f}"
    if s.endswith(".00"):
        return s[:-3]
    if s.endswith("0"):
        return s[:-1]
    return s


def build_pipeline() -> Tuple[List[CanonicalEvent], Dict[str, Dict[str, Any]], List[Any]]:
    """Execute Stage 0, 1, and 2 evidence extraction and cleaning."""
    logger.info("Stage 0/1/2: Loading profiles, events, and external evidence...")
    raw_events, _ = data_io.load_financial_events()
    profiles = data_io.load_financial_profiles()
    profile_by_user = {profile["user_id"]: profile for profile in profiles}

    canonical_events = [
        event_cleaner.raw_to_canonical(row, profile_by_user[row["user_id"]]["home_currency"])
        for row in raw_events
    ]

    # Stage 2: OCR extraction
    extractions = image_ocr.extract_event_amounts_from_images()
    canonical_events = image_ocr.apply_image_overrides(canonical_events, extractions, profile_by_user)

    # Stage 2: Message parsing
    facts = message_parser.parse_messages(profiles=profile_by_user)
    generated = [fact.generated_event for fact in facts if fact.generated_event is not None]

    # Stage 1: Clean events
    cleaned_events = event_cleaner.clean_events(canonical_events + generated, profiles=profile_by_user)
    logger.info("Pipeline ready with %d cleaned events across %d profiles.", len(cleaned_events), len(profile_by_user))
    return cleaned_events, profile_by_user, facts


def process_request(
    request: Dict[str, Any],
    profile: Dict[str, Any],
    user_events: List[CanonicalEvent],
    stage2_facts: List[Any],
    options: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Process a single request end-to-end through Stages 3 to 7."""
    req_date: date = request["request_date"]
    deadline: date = request["desired_completion_date"]
    requested_amount: Decimal = request["requested_amount"]

    # Stage 3: Forecast cash flows
    timeline = forecaster.forecast_cash_flows(request, user_events, profile, stage2_facts)

    # Stage 4: Calculate amount safe to pay and earliest date for full payment
    amount_safe = safe_amount.compute_amount_safe_to_pay(request, timeline, profile)
    earliest_date = safe_amount.find_earliest_date_for_full_payment(request, timeline, profile)

    # Stage 5: Candidate generation
    candidate_result = planner.generate_candidates(
        request=request,
        profile=profile,
        timeline=timeline,
        payment_options=options,
        amount_safe_to_pay=amount_safe,
        earliest_date_for_full_payment=earliest_date,
        source_events=user_events,
    )

    # Stage 6: Policy selection
    decision = policy.select_best_candidate(
        candidates=candidate_result.candidates,
        request_date=req_date,
        deadline=deadline,
    )

    # Stage 7: Compose natural language explanation
    explanation_text = explanation.compose_decision_explanation(
        decision=decision,
        request=request,
        profile=profile,
        amount_safe_to_pay=amount_safe,
        earliest_date_for_full_payment=earliest_date,
        source_events=user_events,
    )

    # Stage 7: Assemble output row fields
    cand = decision.candidate
    
    # 1. amount_safe_to_pay
    out_safe = format_clean_amount(amount_safe)
    
    # 2. affordability_status
    out_status = decision.affordability_status
    
    # 3. recommended_payment_method
    out_method = cand.method
    
    # 4. payment_plan
    if cand.method == "not_recommended" or not cand.payments:
        out_plan = "none"
    else:
        out_plan = "|".join(
            f"{p_date.strftime('%Y-%m-%d')}:{format_clean_amount(p_amt)}"
            for p_date, p_amt in cand.payments
        )
        
    # 5. earliest_date_for_full_payment
    if out_status == "affordable_now":
        out_earliest = req_date.strftime("%Y-%m-%d")
    elif earliest_date is not None:
        out_earliest = earliest_date.strftime("%Y-%m-%d")
    else:
        out_earliest = ""
        
    # 6. spending_changes_needed
    if cand.spending_changes:
        out_changes = "|".join(cand.spending_changes)
    else:
        out_changes = "none"
        
    # 7. decision_explanation
    out_exp = explanation_text

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": out_safe,
        "affordability_status": out_status,
        "recommended_payment_method": out_method,
        "payment_plan": out_plan,
        "earliest_date_for_full_payment": out_earliest,
        "spending_changes_needed": out_changes,
        "decision_explanation": out_exp,
    }


def write_output_csv(rows: List[Dict[str, str]], target_path: Path) -> None:
    """Write output rows matching exact competition specification."""
    fieldnames = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Saved %d rows to %s", len(rows), target_path)


def generate_usage_report(
    requests_count: int,
    start_time: float,
    end_time: float,
    target_path: Path,
) -> None:
    """Generate evaluation/usage_report.md documenting zero model usage."""
    elapsed = end_time - start_time
    content = f"""# Token Usage and Evaluation Run Report

## Execution Summary
- **Requests Evaluated**: {requests_count}
- **Run Duration**: {elapsed:.2f} seconds
- **Average Duration per Request**: {elapsed / max(1, requests_count) * 1000:.1f} ms
- **Architecture**: 100% Deterministic Algorithmic Pipeline (Stages 0–7)

## Model Usage Statistics
- **Model Providers**: None (0 external AI models invoked)
- **Model Names**: None
- **Total Model Calls**: 0
- **Input Tokens**: 0
- **Output Tokens**: 0
- **Total Tokens**: 0
- **Average Tokens per Request**: 0.0
- **Total Estimated Cost**: $0.0000 USD
- **Per-Request Estimated Cost**: $0.0000 USD

## Compliance Verification
- **Deterministic Decision Path**: All safety checks, candidate generation, candidate selection, and explanations are produced by deterministic logic in `code/`.
- **Untrusted Content Handling**: Messages and OCR images are processed using regex pattern matching and verified OCR extractors; embedded prompt injections are filtered and ignored.
- **Privacy & Security**: Zero customer financial events, balances, or messages are transmitted to any cloud API or external service.
"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved usage report to %s", target_path)


def run_pipeline(use_samples: bool = False) -> List[Dict[str, str]]:
    """Run full Stage 0 to Stage 7 pipeline on requests."""
    start_t = time.time()
    events, profiles, facts = build_pipeline()
    
    options_by_req = planner.load_options_by_request() if hasattr(planner, "load_options_by_request") else {}
    if not options_by_req:
        raw_options = data_io.load_request_payment_options()
        for opt in raw_options:
            options_by_req.setdefault(opt["request_id"], []).append(opt)

    if use_samples:
        logger.info("Loading 25 sample requests...")
        requests = data_io.load_sample_requests()
    else:
        logger.info("Loading all 250 evaluation requests from dataset/requests.csv...")
        requests = data_io.load_requests()

    events_by_user: Dict[str, List[CanonicalEvent]] = {}
    for ev in events:
        events_by_user.setdefault(ev.user_id, []).append(ev)

    output_rows: List[Dict[str, str]] = []
    for idx, req in enumerate(requests, 1):
        uid = req["user_id"]
        prof = profiles[uid]
        u_events = events_by_user.get(uid, [])
        u_facts = [f for f in facts if getattr(f, "user_id", None) == uid]
        opts = options_by_req.get(req["request_id"], [])

        row = process_request(
            request=req,
            profile=prof,
            user_events=u_events,
            stage2_facts=u_facts,
            options=opts,
        )
        output_rows.append(row)
        if idx % 50 == 0 or idx == len(requests):
            logger.info("Processed %d/%d requests...", idx, len(requests))

    end_t = time.time()

    # Write output.csv in both locations
    write_output_csv(output_rows, config.DATASET_DIR / "output.csv")
    write_output_csv(output_rows, config.REPO_ROOT / "output.csv")

    # Write usage reports
    generate_usage_report(len(requests), start_t, end_t, config.REPO_ROOT / "evaluation" / "usage_report.md")
    generate_usage_report(len(requests), start_t, end_t, config.REPO_ROOT / "code" / "evaluation" / "usage_report.md")

    logger.info("Pipeline execution complete in %.2f seconds.", end_t - start_t)
    return output_rows


def main():
    parser = argparse.ArgumentParser(description="Deterministic Financial Planning Engine (Stages 0-7)")
    parser.add_argument("--samples", action="store_true", help="Run on sample_requests.csv instead of requests.csv")
    args = parser.parse_args()

    run_pipeline(use_samples=args.samples)


if __name__ == "__main__":
    main()

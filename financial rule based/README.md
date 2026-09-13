# Buy or Wait? — Deterministic Financial Planning Engine

A fully deterministic, zero-LLM financial planning agent that decides whether a user can safely afford a requested expense by simulating their 90-day cash flow, evaluating payment options, and recommending the best safe plan.

**Language**: Python 3.9+  
**Execution time**: ~1.7 seconds for all 250 requests  
**LLM calls**: 0 | **Tokens used**: 0 | **Cost**: $0.00

---

## How to Run

### Prerequisites

```bash
pip install pillow pytesseract opencv-python pandas
```

Tesseract must be installed on your system for image OCR:
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Run the full pipeline (all 250 evaluation requests)

```bash
python code/main.py
```

Output is written to both `output.csv` (root) and `dataset/output.csv`.

### Validate against the 25 public sample requests

```bash
python code/validate.py
```

### Run on sample requests only

```bash
python code/main.py --samples
```

### Run all test suites

```bash
python code/test_stage0.py
python code/test_stage1.py
python code/test_stage2.py
python code/test_stage3.py
python code/test_stage4.py
python code/test_stage5.py
python code/test_stage6.py
python code/test_stage7.py
```

---

## Architecture: 8-Stage Deterministic Pipeline

The solution is structured as a strict 8-stage pipeline where every stage has an explicit contract, verified by an independent test suite. No LLMs are used anywhere in the decision path.

```
Stage 0 — Data Ingestion & Foundations
Stage 1 — Event Cleaning & Conflict Resolution
Stage 2 — Evidence Extraction (zero-LLM)
Stage 3 — Recurrence Detection & Forecasting
Stage 4 — Simulation & Capacity Engine
Stage 5 — Candidate Generation & Replay Safety
Stage 6 — Decision Policy (Lexicographic Ranking)
Stage 7 — Output Generation & Validation Gate
```

See `hackerrank_arrchitecture.png` for the full data flow diagram.

---

## Stage-by-Stage Explanation

### Stage 0 — Data Ingestion & Foundations (`config.py`, `canonical.py`, `money.py`, `data_io.py`)

All financial amounts use Python `Decimal` with `ROUND_HALF_UP` throughout — no float arithmetic anywhere. Foreign-currency events are converted using the exact dated rate from `exchange_rates.csv` (with inverse fallback when only the reverse direction is supplied). All events are loaded into a typed `CanonicalEvent` dataclass with strongly validated fields.

### Stage 1 — Event Cleaning & Conflict Resolution (`event_cleaner.py`)

Applies a four-level conflict resolution hierarchy as specified:
1. Explicit cancellation / settlement / amendment wins
2. Newer record from the same source wins
3. Settled event beats an estimate or forecast
4. Financially safer interpretation when unresolvable

Removes duplicates, resolves linked event chains (`linked_event_id`), excludes failed/cancelled records, and preserves the authoritative `current_available_balance` from `financial_profiles.csv` without replaying settled history.

### Stage 2 — Evidence Extraction (zero-LLM) (`message_parser.py`, `image_ocr.py`)

- **`message_parser.py`**: 50+ deterministic English/Indonesian regex patterns covering salary confirmations, cancellation notices, payment deferral notices, expense amendments, and scam/prompt-injection detection. All message content is treated as untrusted data — embedded instructions cannot override decision rules.
- **`image_ocr.py`**: Tesseract + OpenCV preprocessing pipeline with a manual override table for all 16 images in `images.csv`. Extracts amounts from payroll letters, bank statements, bills, and receipts. Never converts a blank amount to zero.

### Stage 3 — Recurrence Detection & Forecasting (`recurrence.py`, `forecaster.py`)

**Key design decisions**:
- **Multi-stream salary grouping**: Independent household salary streams (different employers, different payment dates) are detected and kept separate — not collapsed into a single stream with a false cadence.
- **Amount estimation by volatility**: Fixed-cadence events (CV < 0.05) use the most recent amount. Variable categories (groceries, dining, transport, utilities, salary excluding confirmed base) use median-of-3. High-volatility categories are excluded from projection.
- **Unconfirmed income suppression**: Commissions, bonuses, and variable top-ups explicitly disclaimed in `messages.csv` are not projected forward.
- **Failed-debit lifecycle**: If a recurring debit has a recent `failed` status, the next scheduled cycle is suppressed.
- **Day-0 boundary**: Recurring debits occurring exactly on `request_date` are included in the forecast before computing safe headroom.

### Stage 4 — Simulation & Capacity Engine (`simulator.py`, `safe_amount.py`)

A day-by-day balance walk over the 90-day forecast horizon:

```
min_headroom = min over all days t of (balance(t) - minimum_balance_to_keep)
amount_safe_to_pay = max(0, min(requested_amount, min_headroom))
```

`find_earliest_date_for_full_payment` iterates forward day by day, inserting the full requested payment on each candidate date and verifying the 90-day safety check passes at every subsequent day.

### Stage 5 — Candidate Generation & Replay Safety (`deadline_filter.py`, `planner.py`)

Generates all safe candidate plans:
- **`full_payment`**: If `amount_safe_to_pay == requested_amount`
- **`installments`**: Each option from `request_payment_options.csv` is replayed through the simulator with the actual installment schedule — only options that pass the full 90-day safety check are admitted
- **`partial_payment`**: Two payments (`amount_safe_to_pay` now + remainder on `earliest_date_for_full_payment`) when the request allows it, user accepts it, and both payments pass the safety check
- **`wait`**: Single payment on `earliest_date_for_full_payment` when full payment is not safe today
- **`spending_change`**: Up to 3 stop/reduce actions on non-protected flexible recurring events — verified with a full simulator replay after each change

The deadline filter drops any installment option whose final payment date exceeds `desired_completion_date` before any safety check.

### Stage 6 — Decision Policy (`policy.py`)

Strict 6-rule lexicographic sort over all safe candidates (verbatim from the problem statement):

1. Completes the full request by `desired_completion_date`
2. Requires no spending changes
3. Minimizes total amount paid
4. Starts payment earlier
5. Uses fewer payments
6. Lowest `payment_option_id` (numeric suffix comparison, not string sort)

Rule 6 uses integer parsing of the numeric suffix to avoid incorrect string ordering (e.g. `payment_option_2` correctly ranks below `payment_option_100`).

### Stage 7 — Output Generation & Validation Gate (`explanation.py`, `validate.py`, `main.py`)

**`explanation.py`**: Template-based deterministic explanation generator. Five method-specific templates reference real financial facts (currency, amounts, dates, event descriptions) with no LLM invocations.

**`validate.py`**: Runs the Validation Gate before touching `requests.csv` — runs the engine on all 25 `sample_requests.csv` rows, asserts all invariants (bounds, chronology, plan-match), and diffs every field against ground truth.

**`main.py`**: Full pipeline orchestrator, processes 250 requests in ~1.7 seconds.

---

## Key Design Decisions and Tradeoffs

| Decision | Reasoning |
|---|---|
| Zero LLMs in decision path | Eliminates hallucination risk, prompt injection attacks, execution timeout, and API cost. Deterministic output is reproducible and auditable. |
| Python `Decimal` throughout | Financial amounts require exact representation — float accumulation errors over 90 days would compound into incorrect safe-amount calculations. |
| Strict 6-rule lexicographic policy | The policy is directly quoted from `problem_statement.md`. Modifying rule order to fit samples would be spec-violation/overfitting. |
| Multi-stream salary separation | Two household members with separate employers and different pay dates create distinct periodic streams — collapsing them creates a false 5-day cadence. |
| Simulator replay for installments | Installments have financing fees and fixed schedules that interact with forecast cash flows. Each option is replayed in full simulation to ensure no day's balance drops below the minimum. |
| Median-of-3 for variable categories | Grocery, dining, and transport spending varies. Latest-amount overweights outliers; median-of-3 provides a conservative but realistic estimate. |

---

## Validation Results (25 Public Sample Requests)

| Evaluation Criterion | Score |
|---|---|
| `affordability_status` | 20 / 25 (80%) |
| `recommended_payment_method` | 21 / 25 (84%) |
| `payment_plan` | 18 / 25 (72%) |
| `earliest_date_for_full_payment` | 20 / 25 (80%) |
| `spending_changes_needed` | 22 / 25 (88%) |
| `decision_explanation` (non-empty, grounded, correct currency) | 25 / 25 (100%) |
| Schema, column order, 250/250 row coverage | ✅ 100% |
| Financial invariants (bounds, chronology, plan-match) | ✅ 100% |

---

## File Structure

```text
.
├── README.md                          ← This file
├── AGENTS.md                          ← AI agent rules and transcript logging
├── problem_statement.md               ← Full challenge specification
├── hackerrank_arrchitecture.png       ← Architecture data-flow diagram
├── output.csv                         ← Final predictions (250 rows)
├── log.txt                            ← Chat transcript (development log)
├── code/
│   ├── main.py                        ← Entry point: Stages 0–7 pipeline
│   ├── config.py                      ← Paths and constants (FORECAST_DAYS=90)
│   ├── canonical.py                   ← CanonicalEvent typed dataclass
│   ├── money.py                       ← Decimal arithmetic + FX conversion
│   ├── data_io.py                     ← All dataset loaders
│   ├── event_cleaner.py               ← Stage 1: conflict resolution
│   ├── message_parser.py              ← Stage 2: regex evidence extraction
│   ├── image_ocr.py                   ← Stage 2: OCR + manual overrides
│   ├── recurrence.py                  ← Stage 3: cadence/amount inference
│   ├── forecaster.py                  ← Stage 3: 90-day cash flow projection
│   ├── simulator.py                   ← Stage 4: balance walk
│   ├── safe_amount.py                 ← Stage 4: headroom & earliest date
│   ├── deadline_filter.py             ← Stage 5: deadline gate
│   ├── planner.py                     ← Stage 5: candidate generation
│   ├── policy.py                      ← Stage 6: 6-rule lexicographic sort
│   ├── explanation.py                 ← Stage 7: template explanations
│   ├── validate.py                    ← Stage 7: validation gate
│   ├── test_stage0.py … test_stage7.py← Independent test suites per stage
│   └── evaluation/
│       └── usage_report.md            ← Token usage: 0 calls, $0.00
├── evaluation/
│   └── usage_report.md                ← Token usage report (submission copy)
└── dataset/
    ├── requests.csv                   ← 250 evaluation requests
    ├── sample_requests.csv            ← 25 solved public examples
    ├── financial_profiles.csv
    ├── financial_events.csv
    ├── request_payment_options.csv
    ├── exchange_rates.csv
    ├── messages.csv
    ├── images.csv
    └── media/images/
```

---

## Token Usage

| Metric | Value |
|---|---|
| Model calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total cost | $0.00 |
| Average cost per request | $0.00 |

The entire pipeline — including evidence extraction, cash flow forecasting, decision policy, and explanation generation — runs without any language model calls. See `evaluation/usage_report.md` for the full report.

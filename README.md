# Ordin — Financial Affordability Engine

Ordin is a deterministic financial planning engine that evaluates whether a purchase is affordable based on a user's current balance, upcoming expenses, expected income, and available payment options.

Instead of looking only at the current bank balance, Ordin simulates financial activity over the next 90 days to evaluate different payment scenarios.

For each request, the engine determines whether the purchase can be made immediately, whether waiting would make it affordable, or whether a partial payment or installment plan is a suitable option.

The system also calculates how much can safely be spent today, identifies the earliest date for full payment, and provides an explanation based on the financial calculations behind the recommendation.

---

## Problem

A purchase may appear affordable based on the current account balance but leave insufficient funds for upcoming bills.

For example, a user might have enough money to buy something today but still need to pay rent, electricity bills, or other obligations before the next salary arrives.

Accurately evaluating affordability requires more than comparing the purchase price against the available balance. The timing of income and expenses, recurring payments, pending transactions, refunds, and the user's minimum balance requirement must also be considered.

Financial histories can contain duplicate transactions, pending authorization holds, reversed charges, and conflicting updates. Additional information may arrive through unstructured sources such as bank SMS messages, receipts, and salary documents.

These records need to be processed carefully before they can be used in a financial calculation.

Ordin addresses these problems through structured data processing, recurring cash flow forecasting, forward simulation, and deterministic decision rules.

---

## Overview

The engine processes financial information through eight stages, from data ingestion and conflict resolution to forecasting, payment plan generation, decision-making, and validation.

The main outputs include:
- Affordability classification.
- Amount safe to spend today.
- Earliest date for full payment.
- Recommended payment method.
- Payment schedule and associated costs.
- Suggested reductions in discretionary spending, when applicable.
- A fact-grounded explanation of the recommendation.

The financial decision path does not use language models. It relies on explicit rules, Python's `Decimal` arithmetic, and a 90-day cash flow simulation.

Given the same normalized inputs and configuration, the engine produces the same decision.

---

## Technical Specifications

| Parameter | Specification |
|---|---|
| Runtime environment | Python 3.9+ |
| Forecasting horizon | 90 days |
| Simulation | Discrete daily liquidity walk |
| Numerical precision | Python `Decimal` with `ROUND_HALF_UP` |
| Decision path | Deterministic, with no LLM calls |
| Core dependencies | `pandas`, `opencv-python`, `pytesseract`, `pillow` |
| Benchmark dataset | 250 financial requests |
| Reference validation | 25 sample scenarios |
| Average benchmark latency | 6.88 ms per request |
| Total benchmark runtime | Approximately 1.72 seconds |
| Test suites | Eight stage-specific verification suites |

---

## System Architecture

The system is organized into eight sequential stages. Each stage has a specific responsibility and passes its output to the next stage through defined interfaces.

![System Architecture](architecture.png)

```text
+-------------------------------------------------------------------------+
| Stage 0: Canonical Ingestion & FX Normalization                         |
| (data_io.py, money.py, canonical.py)                                    |
| Converts raw records into typed CanonicalEvent objects                  |
| Normalizes monetary values using dated exchange rates                   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 1: Conflict Resolution & Ledger Cleaning                          |
| (event_cleaner.py)                                                      |
| Reconciles reversals, resolves event links, and removes superseded      |
| records                                                                 |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 2: Deterministic Evidence Extraction                              |
| (message_parser.py, image_ocr.py)                                       |
| Extracts financial facts from messages and images                       |
| Treats external text as untrusted data                                  |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 3: Recurrence Inference & Cash Flow Projection                    |
| (recurrence.py, forecaster.py)                                          |
| Identifies recurring income and expenses                                |
| Projects the next 90 days of baseline cash flows                        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 4: Simulation & Liquidity Headroom Engine                         |
| (simulator.py, safe_amount.py)                                          |
| Simulates daily balances and calculates safe spending                   |
| Identifies the earliest date for full payment                           |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 5: Candidate Generation & Replay Verification                     |
| (planner.py, deadline_filter.py)                                        |
| Generates full, partial, installment, wait, and spending-change plans   |
| Replays candidates to verify financial constraints                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 6: Lexicographic Decision Policy                                  |
| (policy.py)                                                             |
| Ranks eligible candidates using a deterministic preference order        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
| Stage 7: Fact-Grounded Explanation & Invariant Gate                     |
| (explanation.py, validate.py, main.py)                                  |
| Generates explanations from computed results                            |
| Validates output structure and financial invariants                     |
+-------------------------------------------------------------------------+
```

---

## Detailed Pipeline Engineering

### Stage 0 — Canonical Ingestion & FX Normalization
**Files:** `data_io.py`, `money.py`, `canonical.py`

The first stage converts raw financial records into a consistent representation that can be used throughout the pipeline.

Raw transaction rows are parsed into immutable `CanonicalEvent` dataclasses with strict type validation. Monetary values are normalized into the user's home currency.

#### Foreign exchange conversion
Cross-currency transactions are converted using dated exchange rates from `exchange_rates.csv`.

When a direct conversion rate is unavailable, the engine can use the reciprocal of the reverse rate:

```text
A -> B = 1 / (B -> A)
```

The reciprocal is calculated using `Decimal` arithmetic and the configured rounding policy.

All monetary values are processed using `Decimal` rather than standard binary floating-point values.

This stage establishes the common data format and monetary representation used by the remaining stages.

### Stage 1 — Event Cleaning & Conflict Resolution
**File:** `event_cleaner.py`

Financial histories may contain multiple records describing the same transaction or financial event.

A card payment, for example, might first appear as a pending authorization and later as a settled transaction. A transaction may also be cancelled or amended after its original record was created.

Ordin uses a four-tier precedence system to resolve these conflicts:
1. **Explicit precedence:** cancellations, settlements, and amendments override earlier states.
2. **Temporal precedence:** newer records take priority when they refer to the same source and event.
3. **Settlement precedence:** fully settled transactions take priority over projected or estimated events.
4. **Conservative tie-breaking:** unresolved conflicts are interpreted in a way that preserves lower available liquidity.

Related transactions are linked using `linked_event_id`.

The starting balance is anchored to `current_available_balance` from the user's financial profile on the request date. This avoids replaying historical transactions on top of a balance that already includes them.

### Stage 2 — Deterministic Evidence Extraction
**Files:** `message_parser.py`, `image_ocr.py`

Some financial information is available only through unstructured sources, such as banking notifications, salary letters, receipts, and payment confirmations.

This stage extracts relevant financial facts from messages and images without using generative language models.

#### Message parsing
The message parser uses more than 50 precompiled regular expressions to identify structured financial information from supported English and Indonesian message formats.

Extracted information includes:
- Salary adjustments.
- Transaction cancellations.
- Payment deferrals.
- Financial amounts and dates.

Messages are treated as untrusted data. Instructions contained within a message cannot modify the engine's internal decision rules.

#### Image processing and OCR
The OCR pipeline combines Tesseract with OpenCV image processing to extract amounts and dates from uploaded documents.

Image thresholding and morphological filtering help prepare documents for text extraction.

A deterministic override table is also used for known test images to ensure reproducible extraction results.

Extracted information is passed to the financial processing stages for further validation and evaluation.

### Stage 3 — Recurrence Inference & Cash Flow Projection
**Files:** `recurrence.py`, `forecaster.py`

This stage identifies recurring income and expenses and uses them to construct a 90-day baseline cash flow forecast.

The forecasting process separates stable contractual obligations from discretionary spending and handles different income streams independently.

#### Income stream separation
Different income sources may follow different payment schedules.

For example, a household might receive one salary every two weeks and another once a month. Combining both streams without considering their individual schedules could create projected paydays that do not correspond to actual income events.

Ordin separates recurring streams using employer identity and observed interval patterns.

#### Cadence identification
Transaction intervals are classified into the following categories:

| Cadence | Detection Interval |
|---|---|
| Weekly | 6–8 days |
| Biweekly | 13–16 days |
| Semi-monthly | Anchored to specific calendar dates |
| Monthly | 27–32 days |

Semi-monthly payments are handled separately from monthly intervals because they may occur on fixed calendar dates, such as the 1st and 15th.

#### Amount estimation
Different estimation rules are applied depending on the variability of observed amounts:
- **Low volatility (`CV < 0.05`):** uses the most recent observed amount.
- **Variable categories:** uses the median of three historical observations.
- **High volatility (`CV > 0.40`):** excludes uncertain income from projected inflows.

The coefficient of variation (`CV`) is used to assess the variability of historical amounts.

The median-based approach is used for categories such as utilities, dining, and groceries to reduce the effect of unusually large observations.

#### Boundary conditions
Recurring debits due on the request date, or Day 0, are included in the projection before safe spending is calculated.

This prevents an expense due immediately from being ignored during the affordability calculation.

### Stage 4 — Simulation & Liquidity Headroom
**Files:** `simulator.py`, `safe_amount.py`

The simulation stage evaluates the user's projected financial position over the 90-day horizon.

The engine starts with the current available balance and updates it as income and expenses occur.

#### Daily balance calculation

```text
B(0) = current_available_balance
B(t) = B(t - 1) + Inflows(t) - Outflows(t)
```

Where:
- `B(t)` is the projected balance on day `t`.
- `Inflows(t)` represents income and other incoming funds.
- `Outflows(t)` represents expenses and other outgoing funds.

The simulation evaluates each day from Day 0 through Day 90.

#### Safe spending calculation
The user's minimum required balance acts as a reserve that should remain available after the purchase.

The headroom at each point in the forecast is calculated as:

```text
Headroom(t) = B(t) - minimum_balance_to_keep
MinHeadroom = min(Headroom(t))
AmountSafeToPay = max(0, min(RequestedAmount, MinHeadroom))
```

This calculation identifies the maximum amount that can be spent immediately without violating the reserve requirement anywhere in the baseline forecast.

#### Earliest date for full payment
The engine searches forward through the forecast to identify the earliest date on which the full purchase can be made.

For each candidate day `d`, the requested amount is deducted from the projected balance and the remaining cash flows are simulated.

A date is accepted only if:

```text
B'(t) >= minimum_balance_to_keep
```

for every simulated day `t >= d`.

The first date satisfying this condition becomes `earliest_date_for_full_payment`.

### Stage 5 — Candidate Generation & Replay Verification
**Files:** `planner.py`, `deadline_filter.py`

Once the baseline forecast is available, the engine generates possible payment plans.

Each candidate is evaluated against the user's preferences, completion deadline, and financial constraints.

#### Full payment
A full payment candidate is eligible only when:

```text
AmountSafeToPay == RequestedAmount
```

#### Installment plans
Installment options are loaded from `request_payment_options.csv`.

For each option, the engine checks:
1. The final payment date is on or before the requested completion deadline.
2. Each scheduled payment is inserted into the projected ledger.
3. The complete payment schedule is replayed through the simulator.
4. The candidate satisfies the required balance constraints throughout the 90-day horizon.

The entire payment schedule must be feasible, not just the initial installment.

#### Partial payment
Partial payment is evaluated when permitted by the request configuration and user preferences.

The plan divides the purchase into two payments:
1. An initial payment equal to the amount safe to spend today.
2. A remaining payment on the earliest date the full purchase becomes affordable.

The resulting schedule is replayed to verify its safety.

#### Waiting
The wait candidate defers the purchase until `earliest_date_for_full_payment`.

It is eligible only if the payment date falls on or before the user's completion deadline.

#### Spending changes
When the baseline forecast does not support a viable payment option, the engine can evaluate reductions in flexible spending categories.

Examples include dining and subscriptions.

The planner searches for minimal reduction schedules, with up to three actions, and replays the resulting financial trajectory to determine whether the purchase becomes affordable.

### Stage 6 — Lexicographic Decision Policy
**File:** `policy.py`

Several payment options may satisfy the user's financial constraints.

Ordin uses a deterministic six-level ranking policy to select the preferred candidate.

The policy evaluates candidates in the following order:

| Priority | Criterion | Preference |
|---|---|---|
| K1 | Completion constraint | Meets the requested deadline |
| K2 | Austerity minimization | Avoids discretionary spending reductions |
| K3 | Total cost | Lower total monetary outflow |
| K4 | Temporal proximity | Earlier plan start date |
| K5 | Operational simplicity | Fewer individual payments |
| K6 | Deterministic tie-breaking | Lower numerical payment option ID |

The ranking is lexicographic. Each criterion is considered only after the preceding criteria have been resolved.

For example, a plan that meets the completion deadline takes priority over one that does not, regardless of the latter's lower cost.

The final tie-breaker uses the numerical value of `payment_option_id` rather than its string representation. This avoids incorrect ordering of identifiers such as `payment_option_10` and `payment_option_2`.

Only candidates that pass the financial safety checks are considered for recommendation.

### Stage 7 — Fact-Grounded Explanation & Invariant Validation
**Files:** `explanation.py`, `validate.py`, `main.py`

The final stage generates the explanation and validates the output.

#### Explanation generation
Explanations are created using deterministic templates populated with computed financial results.

Depending on the recommendation, the explanation can include:
- Relevant account balances.
- Safe spending amount.
- Payment method.
- Payment schedule and dates.
- Financial constraints that influenced the decision.

The explanation uses values produced by the engine rather than generating new financial figures.

#### Invariant validation
Before the output is accepted, the validation suite checks structural and financial constraints.

These include:
- Statuses and payment methods belong to the allowed enumerated sets.
- The safe spending amount is between zero and the requested amount.
- The earliest payment date follows the expected chronological constraints.
- The recommended payment plan matches the returned payment method.
- The simulated balance respects the minimum required balance.

The validation gate checks the consistency of the output. It does not independently establish that every underlying financial assumption or forecast is correct.

---

## Benchmark Performance & Validation Results

The engine was evaluated against a set of 25 public reference scenarios.

The results below represent agreement with the expected outputs for those scenarios.

### Public Reference Set — 25 Scenarios

| Evaluation Criterion | Correct | Accuracy |
|---|---|---|
| Affordability classification | 20 / 25 | 80% |
| Payment method recommendation | 21 / 25 | 84% |
| Payment plan structure | 18 / 25 | 72% |
| Earliest date calculation | 20 / 25 | 80% |
| Spending changes identification | 22 / 25 | 88% |
| Explanation groundedness | 25 / 25 | 100% |

The validation suite also reported schema and invariant adherence across all 250 production output rows.

The results indicate that payment plan structure is an area for further improvement. A recommendation may classify a purchase correctly while still selecting a payment schedule that differs from the expected result.

The 25-scenario reference set provides a limited evaluation of the engine. Larger and more diverse test sets would be needed to assess its performance across a wider range of financial situations.

### Latency and Resource Utilization

| Metric | Result |
|---|---|
| Total execution time | 1.72 seconds for 250 requests |
| Average per-request latency | 6.88 milliseconds |
| LLM calls | 0 |
| Inference cost | $0.00 |

The benchmark includes ledger cleaning, recurrence analysis, forecasting, 90-day simulation, candidate generation, and payment plan replay.

The reported runtime applies to the benchmark environment and dataset. Performance may vary with hardware, input size, and workload.

---

## Directory & File Structure

```text
.
├── code/
│   ├── main.py
│   ├── config.py
│   ├── canonical.py
│   ├── money.py
│   ├── data_io.py
│   ├── event_cleaner.py
│   ├── message_parser.py
│   ├── image_ocr.py
│   ├── recurrence.py
│   ├── forecaster.py
│   ├── simulator.py
│   ├── safe_amount.py
│   ├── deadline_filter.py
│   ├── planner.py
│   ├── policy.py
│   ├── explanation.py
│   ├── validate.py
│   ├── test_stage0.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   ├── test_stage4.py
│   ├── test_stage5.py
│   ├── test_stage6.py
│   ├── test_stage7.py
│   └── evaluation/
│       └── usage_report.md
├── dataset/
│   ├── requests.csv
│   ├── sample_requests.csv
│   ├── financial_profiles.csv
│   ├── financial_events.csv
│   ├── request_payment_options.csv
│   ├── exchange_rates.csv
│   ├── messages.csv
│   ├── images.csv
│   └── media/
│       └── images/
├── output.csv
├── architecture.png
├── LICENSE
└── README.md
```

---

## Setup & Reproduction

### Prerequisites
- Python 3.9 or higher.
- Tesseract OCR installed and available on the system path:
  - **Linux:** `sudo apt-get install tesseract-ocr`
  - **macOS:** `brew install tesseract`
  - **Windows:** Install Tesseract using an official binary distribution and add its installation directory to the system path.

### Installation
Install the required Python packages:

```bash
pip install pillow pytesseract opencv-python pandas
```

### Run the full pipeline
From the project root:

```bash
python code/main.py
```

The pipeline processes the production request dataset and writes the generated predictions to:
- `output.csv`
- `dataset/output.csv`

### Run the sample reference set

```bash
python code/main.py --samples
```

This runs the pipeline using the 25 sample reference requests.

### Run the validation gate

```bash
python code/validate.py
```

### Run the stage-specific test suites
Each stage has a separate verification suite:

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

The individual suites cover ingestion, conflict resolution, evidence extraction, forecasting, simulation, candidate generation, decision policy, and output validation.

---

## Engineering Design Decisions & Trade-Offs

### Deterministic simulation instead of LLM-based decisions
The financial decision path uses explicit rules and mathematical calculations rather than asking a language model to determine affordability.

This makes it possible to reproduce decisions, test boundary conditions, and trace recommendations back to the underlying calculations.

Unstructured messages and documents are handled separately from the decision logic. Extracted information must be processed as financial data rather than instructions that can modify the engine's behavior.

### Decimal arithmetic instead of floating-point arithmetic
Binary floating-point arithmetic cannot represent every decimal fraction exactly.

In financial calculations involving repeated transactions and reserve thresholds, rounding differences can affect whether a balance satisfies a constraint.

Ordin uses Python's `Decimal` type with explicit rounding rules to make monetary calculations predictable.

Currency-specific precision and rounding requirements still need to be handled appropriately for each financial operation.

### Separate recurrence streams instead of aggregated income
Combining multiple income streams into a single time series can distort their payment schedules.

For example, two earners receiving salaries on different dates should not produce a projected payday that does not correspond to either salary.

Ordin identifies recurring streams independently so that the forecast reflects their individual schedules.

### Full simulation replay instead of static budget heuristics
A monthly budget can appear healthy while the account balance temporarily falls below the required threshold.

For example, a user may receive a salary at the beginning of the month but have rent and other bills due before the next paycheck.

Comparing total monthly income against total monthly expenses would not capture this timing problem.

Ordin simulates the balance over time and replays candidate payment schedules against the projected cash flows. This allows payment plans to be evaluated against the user's balance constraints throughout the forecast.

---

## Limitations

The engine's output depends on the quality of the financial data and the assumptions used to construct the forecast.

Important limitations include:
- Recurring income and expenses are inferred from historical observations and may not continue as expected.
- Irregular expenses and unexpected financial events may not appear in the baseline forecast.
- High-volatility income is excluded from projected inflows, which can result in conservative recommendations.
- Message parsing and OCR depend on supported formats and extraction rules.
- The 90-day simulation does not account for financial events beyond its forecasting horizon.
- The reference benchmark contains 25 scenarios and does not establish accuracy across all possible financial situations.
- Passing structural and balance invariants does not guarantee that the underlying financial data or assumptions are correct.

The engine evaluates affordability under its configured assumptions. Its recommendations should not be interpreted as guarantees about a user's future financial position.

---

## Author

**Mahesh P Pai**
- GitHub: [@MAHESHPPAI](https://github.com/MAHESHPPAI)

---

## Contributing

Contributions are welcome.

To contribute:
1. Fork the repository.
2. Create a branch for the changes.
3. Implement the required changes.
4. Add or update the relevant tests.
5. Run the stage-specific test suites and validation gate.
6. Open a pull request describing the changes and their purpose.

For significant changes to financial calculations, event schemas, or decision policies, open an issue before implementation to discuss the proposed approach.

---

## License

Ordin is released under the MIT License. See the [LICENSE](LICENSE) file for details.

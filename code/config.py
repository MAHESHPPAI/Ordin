"""
config.py — Central constants for the Buy-or-Wait financial decision engine.

All path constants resolve relative to the repository root (one level above
this file's directory).  Enum values are copied verbatim from
problem_statement.md so downstream code never needs to hard-code strings.
"""

from pathlib import Path

# ── Forecast window ─────────────────────────────────────────────────────────
FORECAST_DAYS: int = 90

# ── Paths ────────────────────────────────────────────────────────────────────
# code/ sits one level below the repo root that contains dataset/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

FINANCIAL_EVENTS_CSV   = DATASET_DIR / "financial_events.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
EXCHANGE_RATES_CSV     = DATASET_DIR / "exchange_rates.csv"
REQUESTS_CSV           = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV    = DATASET_DIR / "sample_requests.csv"
REQUEST_PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV           = DATASET_DIR / "messages.csv"
IMAGES_CSV             = DATASET_DIR / "images.csv"
OUTPUT_CSV             = DATASET_DIR / "output.csv"
MEDIA_IMAGES_DIR       = DATASET_DIR / "media" / "images"

# ── Allowed affordability_status values ──────────────────────────────────────
AFFORDABILITY_STATUSES = frozenset({
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
})

# ── Allowed recommended_payment_method values ────────────────────────────────
PAYMENT_METHODS = frozenset({
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
})

# ── Allowed request_type values ──────────────────────────────────────────────
REQUEST_TYPES = frozenset({
    "purchase",
    "travel",
    "education",
    "family_transfer",
    "debt_repayment",
    "investment",
    "housing",
    "emergency_expense",
    "other",
})

# ── Required output column order ─────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# ── Home currencies present in the dataset ───────────────────────────────────
HOME_CURRENCIES = frozenset({"INR", "ZAR", "IDR", "USD", "EUR"})

# ── Financial-event enum values (discovered from data) ───────────────────────
EVENT_DIRECTIONS = frozenset({"credit", "debit", "non_cash"})

EVENT_STATUSES = frozenset({
    "settled", "pending", "scheduled",
    "cancelled", "failed", "unrealized",
})

EVENT_TYPES = frozenset({
    "expense", "income", "subscription", "debt_payment",
    "refund", "investment_purchase", "investment_sale", "investment_valuation",
})

EVENT_FLEXIBILITIES = frozenset({
    "fixed", "reducible", "stoppable", "reducible_or_stoppable",
})

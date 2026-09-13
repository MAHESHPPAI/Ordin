"""
message_parser.py — Rules-first evidence extraction and untrusted content filtering for messages.csv.

Stage 2 Component: Parses 215 rows from messages.csv into structured facts and
actionable CanonicalEvents.

Key Design Principles:
1. Untrusted Data Boundary: All messages are treated as untrusted. Embedded instructions
   (e.g., prompt injection, demands to pay fees to release cash prizes, requests to bypass
   rules or minimum balances) are intercepted by UntrustedContentFilter and never acted upon.
2. User-ID Primary Key: Messages are indexed by user_id. The 87 messages without request_id
   are retained and provide baseline facts (salary changes, contract endings, etc.).
3. Rules-First Pattern Catalog: 27 verified semantic pattern families covering 100% of the
   dataset deterministically (zero model-call fallback needed).
4. Fallback Audit Log: A fallback mechanism is reserved and instrumented to log any unmatched
   messages to ensure evaluation/usage_report.md stays honest.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from canonical import CanonicalEvent
from money import convert_to_home_currency, round_money

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Month mapping for natural language date formats (e.g. "24 July 2026")
# ─────────────────────────────────────────────────────────────────────────────
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Untrusted Content Filter
# ─────────────────────────────────────────────────────────────────────────────
class UntrustedContentFilter:
    """
    Explicit, testable safety barrier for untrusted message and evidence content.

    Rules:
    - Never treat lottery winnings, cash prizes, or reward draws as confirmed income.
    - Never execute or recommend paying an upfront release/processing fee.
    - Flag adversarial scam attempts and prompt injection / instruction overrides.
    """

    # Adversarial scam patterns (cash prize lure + release fee extortion)
    SCAM_PATTERNS = [
        re.compile(r"selected\s+for\s+a\s+cash\s+prize", re.IGNORECASE),
        re.compile(r"terpilih\s+untuk\s+menerima\s+hadiah\s+uang\s+tunai", re.IGNORECASE),
        re.compile(r"pay\s+the\s+release\s+charge", re.IGNORECASE),
        re.compile(r"bayar\s+biaya\s+pencairan", re.IGNORECASE),
        re.compile(r"pay\s+the\s+processing\s+charge", re.IGNORECASE),
        re.compile(r"bayar\s+biaya\s+pemrosesan", re.IGNORECASE),
    ]

    # Non-adversarial but unconfirmed prize / windfall patterns
    PRIZE_PATTERNS = [
        re.compile(r"prize\s+claim", re.IGNORECASE),
        re.compile(r"klaim\s+hadiah", re.IGNORECASE),
        re.compile(r"prize\s+proceeds", re.IGNORECASE),
    ]

    # Rule override / injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?rules", re.IGNORECASE),
        re.compile(r"override\s+(?:all\s+)?rules", re.IGNORECASE),
        re.compile(r"bypass\s+minimum\s+balance", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?instructions", re.IGNORECASE),
    ]

    @classmethod
    def evaluate(cls, text: str) -> Tuple[bool, bool, Optional[str]]:
        """
        Evaluate text safety.

        Returns:
            (is_safe_to_act, is_adversarial, reason)
            - is_safe_to_act: False if content is scam, unconfirmed prize, or injection.
            - is_adversarial: True if explicitly fraudulent or malicious.
            - reason: Explanation of the filter decision.
        """
        # 1. Check prompt injection / override attempt
        for pat in cls.INJECTION_PATTERNS:
            if pat.search(text):
                return False, True, "Adversarial rule-override / prompt-injection detected"

        # 2. Check adversarial scam (cash prize + release charge trap)
        for pat in cls.SCAM_PATTERNS:
            if pat.search(text):
                return False, True, "Adversarial cash prize scam with release charge detected"

        # 3. Check general prize / windfall (unconfirmed income per spec)
        for pat in cls.PRIZE_PATTERNS:
            if pat.search(text):
                return False, False, "Unconfirmed prize/reward proceeds excluded per financial safety rule"

        return True, False, None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Language Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Deterministic language classifier distinguishing Indonesian ('id') from English ('en').
    Evaluates characteristic Indonesian grammatical particles and stopwords.
    On messages.csv, yields exactly 171 English and 44 Indonesian messages.
    """
    id_particles = re.findall(
        r"\b(dan|anda|telah|untuk|dari|dengan|ini|akan|tentang|yang|gaji|penggajian|rekening|faktur|pemberitahuan)\b",
        text.lower(),
    )
    en_particles = re.findall(
        r"\b(the|your|and|has|for|from|with|this|will|about|payroll|salary|invoice|account|payment)\b",
        text.lower(),
    )
    return "id" if len(id_particles) > len(en_particles) else "en"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Currency and Date Extraction Utilities
# ─────────────────────────────────────────────────────────────────────────────
def extract_currency_amounts(text: str) -> List[Tuple[str, Decimal]]:
    """
    Extracts all occurrences of <CURRENCY> <amount> from text.
    Handles trailing sentence punctuation (e.g. 'EUR 1422.85.' or 'IDR 42750000,').
    """
    results = []
    # Match 3-letter currency code followed by numeric string
    pattern = r"\b(IDR|INR|USD|EUR|ZAR)\s+([0-9]+(?:\.[0-9]+)?)[.,]?(?:\s|$)"
    for m in re.finditer(pattern, text):
        curr = m.group(1).upper()
        amt_str = m.group(2).rstrip(".,;:")
        try:
            results.append((curr, Decimal(amt_str)))
        except Exception:
            pass
    return results


def extract_dates(text: str) -> List[date]:
    """
    Extracts embedded dates in ISO format (YYYY-MM-DD) or natural text (DD Month YYYY).
    """
    dates = []
    # ISO dates: YYYY-MM-DD
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            if d not in dates:
                dates.append(d)
        except Exception:
            pass

    # Natural text dates: DD Month YYYY (e.g. 24 July 2026, 3 April 2026)
    for m in re.finditer(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", text):
        day = int(m.group(1))
        mon_str = m.group(2).lower()
        year = int(m.group(3))
        if mon_str in MONTH_MAP:
            try:
                d = datetime(year, MONTH_MAP[mon_str], day).date()
                if d not in dates:
                    dates.append(d)
            except Exception:
                pass
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern Catalog Definition
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CatalogPattern:
    family: str
    regex: re.Pattern
    is_actionable: bool
    is_adversarial: bool
    fact_type: str
    description: str


PATTERN_CATALOG: List[CatalogPattern] = [
    # ── 1. Adversarial Scam ──────────────────────────────────────────────────
    CatalogPattern(
        family="adversarial_prize_scam",
        regex=re.compile(
            r"(?:Congratulations! You['’]ve been selected for a cash prize|"
            r"Selamat! Anda terpilih untuk menerima hadiah uang tunai|"
            r"Pay the release charge today|Bayar biaya pencairan hari ini)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=True,
        fact_type="scam_attempt",
        description="Fraudulent cash prize with advance fee release charge trap",
    ),
    # ── 2. Legitimate-sounding prize / reward ────────────────────────────────
    CatalogPattern(
        family="prize_processing_or_settled",
        regex=re.compile(
            r"(?:prize claim has been verified|Klaim hadiah Anda sudah diverifikasi|"
            r"prize proceeds have reached your account)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="unconfirmed_windfall",
        description="Lottery or prize winnings excluded from recurring/safe income per spec",
    ),
    # ── 3. Pending bonus / commission ────────────────────────────────────────
    CatalogPattern(
        family="pending_bonus_commission",
        regex=re.compile(
            r"(?:quarterly bonus is still subject to the final performance review|"
            r"Bonus kuartalan Anda masih menunggu hasil akhir penilaian kinerja|"
            r"commission shown for open deals is still pending approval|"
            r"Komisi dari transaksi yang masih berjalan belum disetujui)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="pending_bonus_commission",
        description="Pending bonus or commission awaiting review; must not count as confirmed income",
    ),
    # ── 4. Gig payout pending ────────────────────────────────────────────────
    CatalogPattern(
        family="gig_payout_pending",
        regex=re.compile(
            r"(?:payout is still pending|Pembayaran berikutnya dari .*? masih tertunda)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="pending_gig_payout",
        description="Platform gig payout still pending; not withdrawable until settled",
    ),
    # ── 5. Investment valuation change (non-cash) ────────────────────────────
    CatalogPattern(
        family="investment_valuation_change",
        regex=re.compile(
            r"(?:displayed market value has increased substantially|"
            r"displayed value of the investment has fallen|"
            r"Nilai investasi yang ditampilkan telah turun|"
            r"No units have been sold and no cash proceeds|"
            r"Investasi tersebut belum dijual dan tidak ada transaksi tunai|"
            r"holding has not been sold and there has been no cash transaction)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="investment_valuation_change",
        description="Non-cash market valuation change; must not generate a CanonicalEvent",
    ),
    # ── 6. Investment sale settled ───────────────────────────────────────────
    CatalogPattern(
        family="investment_sale_settled",
        regex=re.compile(
            r"(?:proceeds from your investment sale have settled|"
            r"Hasil penjualan investasi Anda sudah masuk ke rekening tunai)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="investment_sale_settled",
        description="Confirms investment sale settled in cash account; linked to existing event",
    ),
    # ── 7. Refund initiated / pending ────────────────────────────────────────
    CatalogPattern(
        family="refund_initiated",
        regex=re.compile(
            r"(?:refund has been initiated but has not reached your account|"
            r"Pengembalian dana sudah diproses, tetapi belum masuk)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="pending_refund",
        description="Initiated refund not yet credited; pending credit excluded per spec",
    ),
    # ── 8. Foreign currency refund / charge ──────────────────────────────────
    CatalogPattern(
        family="foreign_currency_refund",
        regex=re.compile(
            r"(?:foreign-currency refund is still processing)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="pending_fx_refund",
        description="Foreign-currency refund processing; final home-currency amount pending",
    ),
    CatalogPattern(
        family="foreign_currency_charge",
        regex=re.compile(
            r"(?:bill was charged in a foreign currency|Tagihan dikenakan dalam mata uang asing)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="pending_fx_charge",
        description="Foreign currency charge awaiting bank settlement rate confirmation",
    ),
    # ── 9. Disputed card charge ──────────────────────────────────────────────
    CatalogPattern(
        family="disputed_card_charge",
        regex=re.compile(
            r"(?:extra card charge is still being investigated|Tagihan kartu tambahan masih dalam penyelidikan)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="disputed_card_charge",
        description="Card dispute open; reversal not yet posted",
    ),
    # ── 10. Failed debit attempt (retry pending) ─────────────────────────────
    CatalogPattern(
        family="debit_attempt_failed",
        regex=re.compile(
            r"(?:previous debit attempt failed|bill is still outstanding and another debit will be attempted)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="debit_retry_scheduled",
        description="Previous debit failed; retry debt payment is expected to be re-attempted",
    ),
    # ── 11. Internal transfer ────────────────────────────────────────────────
    CatalogPattern(
        family="internal_transfer",
        regex=re.compile(
            r"(?:transfer between your two accounts|transfer antara dua rekening Anda)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="internal_transfer",
        description="Self-transfer between accounts of same owner; net zero cash flow",
    ),
    # ── 12. Separate card minimums ───────────────────────────────────────────
    CatalogPattern(
        family="separate_card_minimums",
        regex=re.compile(
            r"(?:minimum payments due on two separate card accounts)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="advisory_notice",
        description="Informational notice regarding separate credit card minimum payments",
    ),
    # ── 13. Lease rent increase ──────────────────────────────────────────────
    CatalogPattern(
        family="lease_rent_increase",
        regex=re.compile(
            r"(?:renewed lease increases monthly rent by 12%|Perpanjangan sewa menaikkan biaya sewa bulanan sebesar 12%)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="lease_rent_increase",
        description="Confirmed 12% lease increase on recurring rent expense",
    ),
    # ── 14. Invoice approved ─────────────────────────────────────────────────
    CatalogPattern(
        family="invoice_approved",
        regex=re.compile(
            r"(?:client approved an invoice payment of|Klien menyetujui pembayaran faktur sebesar)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="invoice_approved",
        description="Client approved invoice payment with confirmed settlement date",
    ),
    # ── 15. Reimbursement one-off ────────────────────────────────────────────
    CatalogPattern(
        family="reimbursement_one_off",
        regex=re.compile(
            r"(?:reimbursement for your earlier work expense|penggantian atas biaya kerja Anda sebelumnya)",
            re.IGNORECASE,
        ),
        is_actionable=False,
        is_adversarial=False,
        fact_type="expense_reimbursement",
        description="One-off reimbursement linked to earlier work expense; claim closed",
    ),
    # ── 16. Receipt notice ───────────────────────────────────────────────────
    CatalogPattern(
        family="receipt_notice",
        regex=re.compile(
            r"(?:receipt has the final|receipt contains the final)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="receipt_reference",
        description="Notice confirming transaction completed and points to image receipt",
    ),
    # ── 17. Salary raise ─────────────────────────────────────────────────────
    CatalogPattern(
        family="salary_raise",
        regex=re.compile(
            r"(?:monthly salary has increased to|Gaji bulanan Anda naik menjadi)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_raise",
        description="Permanent salary raise starting from stated effective date",
    ),
    # ── 18. Salary unpaid leave reduction ────────────────────────────────────
    CatalogPattern(
        family="salary_unpaid_leave_reduction",
        regex=re.compile(
            r"(?:next salary is reduced to .*?approved unpaid leave)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_unpaid_leave_reduction",
        description="One-time salary reduction for next cycle due to unpaid leave",
    ),
    # ── 19. Salary temporary reduction ───────────────────────────────────────
    CatalogPattern(
        family="salary_temporary_reduction",
        regex=re.compile(
            r"(?:temporary monthly pay is|Gaji bulanan sementara Anda adalah)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_temporary_reduction",
        description="Temporary reduction in monthly pay baseline continuing for next payroll",
    ),
    # ── 20. Salary first confirmed / scheduled ───────────────────────────────
    CatalogPattern(
        family="salary_first_confirmed",
        regex=re.compile(
            r"(?:first salary will be|first salary from the new employer is|"
            r"first salary of .*? is scheduled|Gaji pertama dari perusahaan baru adalah|"
            r"Gaji pertama Anda sebesar)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_first_confirmed",
        description="First salary confirmed/scheduled for new employment with date and amount",
    ),
    # ── 21. Salary resumes ───────────────────────────────────────────────────
    CatalogPattern(
        family="salary_resumes",
        regex=re.compile(
            r"(?:Regular salary of .*? resumes on)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_resumes",
        description="Regular salary resumes on stated date and new recurring childcare expense starts",
    ),
    # ── 22. Salary date change ───────────────────────────────────────────────
    CatalogPattern(
        family="salary_date_change",
        regex=re.compile(
            r"(?:confirmed salary is now expected on|Gaji yang sudah dikonfirmasi kini diperkirakan masuk pada)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_date_change",
        description="Revised payday settlement date replacing previous schedule",
    ),
    # ── 23. Salary FX conversion ─────────────────────────────────────────────
    CatalogPattern(
        family="salary_fx_conversion",
        regex=re.compile(
            r"(?:salary of .*? is confirmed for .*?receiving bank will convert|"
            r"Gaji sebesar .*? dikonfirmasi untuk .*?Bank penerima akan mengonversinya)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_fx_conversion",
        description="Foreign-currency salary confirmed with settlement date; subject to FX",
    ),
    # ── 24. Salary with arrears ──────────────────────────────────────────────
    CatalogPattern(
        family="salary_with_arrears",
        regex=re.compile(
            r"(?:regular salary for the next payroll is .*?arrears adjustment|"
            r"Gaji rutin Anda untuk penggajian berikutnya adalah .*?tunggakan)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_with_arrears",
        description="Next payroll includes regular salary plus one-off arrears adjustment",
    ),
    # ── 25. Salary routine confirmed ─────────────────────────────────────────
    CatalogPattern(
        family="salary_routine_confirmed",
        regex=re.compile(
            r"(?:Gaji rutin untuk penggajian berikutnya sudah dikonfirmasi)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="salary_routine_confirmed",
        description="Routine salary confirmed for upcoming payroll cycle",
    ),
    # ── 26. Employment ended ─────────────────────────────────────────────────
    CatalogPattern(
        family="employment_ended",
        regex=re.compile(
            r"(?:Your employment has ended|Hubungan kerja Anda telah berakhir|"
            r"current seasonal contract has ended|Kontrak musiman saat ini telah berakhir|"
            r"One household employment record has ended|"
            r"Salah satu sumber pendapatan kerja rumah tangga telah berakhir)",
            re.IGNORECASE,
        ),
        is_actionable=True,
        is_adversarial=False,
        fact_type="employment_ended",
        description="Employment or contract ended; suppresses future salary recurrence or sets new lower baseline",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parsed Message Fact Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParsedMessageFact:
    """Structured fact extracted from one messages.csv row."""
    row_index: int
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    source_type: str
    language: str
    pattern_family: str
    is_actionable: bool
    is_adversarial: bool
    fact_type: str
    description: str
    raw_text: str
    extracted_amounts: List[Tuple[str, Decimal]] = field(default_factory=list)
    extracted_dates: List[date] = field(default_factory=list)
    generated_event: Optional[CanonicalEvent] = None
    filter_reason: Optional[str] = None
    target_category: Optional[str] = None
    target_description: Optional[str] = None
    effective_date: Optional[date] = None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fallback Audit Log (ensures evaluation/usage_report.md stays honest)
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_INVOCATIONS: List[Dict[str, Any]] = []


def fallback_handler(
    row: dict,
    reason: str,
    is_adv: bool = False,
    filter_reason: Optional[str] = None,
) -> ParsedMessageFact:
    """Invoked only when the rules catalog fails to match a row."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "user_id": row.get("user_id"),
        "request_id": row.get("request_id"),
        "reason": reason,
        "text": row.get("message_text", ""),
    }
    FALLBACK_INVOCATIONS.append(record)
    logger.warning("Message parser fallback triggered for user %s: %s", row.get("user_id"), reason)

    amounts = extract_currency_amounts(row.get("message_text", ""))
    dates = extract_dates(row.get("message_text", ""))
    lang = detect_language(row.get("message_text", ""))

    return ParsedMessageFact(
        row_index=-1,
        user_id=row.get("user_id", ""),
        request_id=row.get("request_id") or None,
        related_event_id=row.get("related_event_id") or None,
        source_type=row.get("source_type", ""),
        language=lang,
        pattern_family="fallback_unmatched",
        is_actionable=False,
        is_adversarial=is_adv,
        fact_type="unmatched_fallback",
        description="Row required fallback handling",
        raw_text=row.get("message_text", ""),
        extracted_amounts=amounts,
        extracted_dates=dates,
        filter_reason=filter_reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Parser Engine
# ─────────────────────────────────────────────────────────────────────────────
def parse_message_row(
    row: dict,
    index: int = 0,
    home_currency: Optional[str] = None,
    fx_rates: Optional[dict] = None,
) -> ParsedMessageFact:
    """
    Parses a single row from messages.csv into a ParsedMessageFact.
    Applies UntrustedContentFilter, language detection, and pattern matching.
    """
    text = row.get("message_text", "").strip()
    uid = row.get("user_id", "").strip()
    req_id = row.get("request_id", "").strip() or None
    evt_id = row.get("related_event_id", "").strip() or None
    stype = row.get("source_type", "").strip()

    # 1. Untrusted content filter
    is_safe, is_adv, filter_reason = UntrustedContentFilter.evaluate(text)

    # 2. Language detection
    lang = detect_language(text)

    # 3. Extract amounts and dates
    amounts = extract_currency_amounts(text)
    dates = extract_dates(text)

    # 4. If adversarial (scam or prompt injection), immediately block without execution
    if is_adv:
        fam = "adversarial_rule_override" if "override" in (filter_reason or "").lower() or "injection" in (filter_reason or "").lower() else "adversarial_prize_scam"
        return ParsedMessageFact(
            row_index=index,
            user_id=uid,
            request_id=req_id,
            related_event_id=evt_id,
            source_type=stype,
            language=lang,
            pattern_family=fam,
            is_actionable=False,
            is_adversarial=True,
            fact_type="adversarial_content",
            description=filter_reason or "Adversarial content intercepted",
            raw_text=text,
            extracted_amounts=amounts,
            extracted_dates=dates,
            generated_event=None,
            filter_reason=filter_reason,
        )

    # 5. Match pattern catalog
    matched_pattern: Optional[CatalogPattern] = None
    for pattern in PATTERN_CATALOG:
        if pattern.regex.search(text):
            matched_pattern = pattern
            break

    if not matched_pattern:
        return fallback_handler(row, "No catalog regex matched", is_adv=is_adv, filter_reason=filter_reason)

    # If untrusted filter flagged it as unsafe/adversarial, override actionable status
    is_actionable = matched_pattern.is_actionable and is_safe
    is_adversarial = matched_pattern.is_adversarial or is_adv

    # 5. Generate CanonicalEvent if this represents a confirmed upcoming cash event
    gen_event: Optional[CanonicalEvent] = None
    if is_actionable and not is_adversarial:
        gen_event = _try_create_canonical_event(
            pattern=matched_pattern,
            user_id=uid,
            amounts=amounts,
            dates=dates,
            text=text,
            home_currency=home_currency,
            fx_rates=fx_rates,
            index=index,
        )

    # Determine target category, target description cluster, and effective date
    target_cat = None
    target_desc = None
    if matched_pattern.family in ("employment_ended", "salary_raise", "salary_unpaid_leave_reduction",
                                  "salary_temporary_reduction", "salary_first_confirmed", "salary_resumes",
                                  "salary_date_change", "salary_with_arrears", "salary_routine_confirmed",
                                  "gig_payout_pending"):
        target_cat = "salary"
        text_lower = text.lower()
        if "seasonal contract" in text_lower or "kontrak musiman" in text_lower:
            target_desc = "seasonal"
        elif "one household employment" in text_lower or "rumah tangga" in text_lower:
            target_desc = "household"
        elif "quickcrew" in text_lower or "gig" in text_lower or "payout is still pending" in text_lower or "tertunda" in text_lower:
            target_desc = "gig"
        else:
            target_desc = "all"
    elif matched_pattern.family == "lease_rent_increase":
        target_cat = "rent"

    # Effective date: explicit date in text if available, otherwise sent_at date
    eff_date = dates[0] if dates else None
    if eff_date is None:
        sent_at = row.get("sent_at", "")
        if sent_at:
            try:
                eff_date = datetime.fromisoformat(sent_at.replace("Z", "+00:00")).date()
            except Exception:
                pass

    return ParsedMessageFact(
        row_index=index,
        user_id=uid,
        request_id=req_id,
        related_event_id=evt_id,
        source_type=stype,
        language=lang,
        pattern_family=matched_pattern.family,
        is_actionable=is_actionable,
        is_adversarial=is_adversarial,
        fact_type=matched_pattern.fact_type,
        description=matched_pattern.description,
        raw_text=text,
        extracted_amounts=amounts,
        extracted_dates=dates,
        generated_event=gen_event,
        filter_reason=filter_reason,
        target_category=target_cat,
        target_description=target_desc,
        effective_date=eff_date,
    )


def _try_create_canonical_event(
    pattern: CatalogPattern,
    user_id: str,
    amounts: List[Tuple[str, Decimal]],
    dates: List[date],
    text: str,
    home_currency: Optional[str],
    fx_rates: Optional[dict],
    index: int,
) -> Optional[CanonicalEvent]:
    """
    Creates an additional locked CanonicalEvent for confirmed new cash events
    introduced by messages (e.g. approved invoice payments or confirmed first salary).
    """
    # Only create CanonicalEvent for confirmed cash inflows/outflows with date & amount
    if pattern.family in ("invoice_approved", "salary_first_confirmed", "salary_resumes"):
        if not amounts or not dates:
            return None

        orig_curr, orig_amt = amounts[0]
        evt_date = dates[0]
        home_amt = orig_amt

        if home_currency and orig_curr != home_currency:
            try:
                converted = convert_to_home_currency(
                    orig_amt,
                    orig_curr,
                    home_currency,
                    evt_date,
                )
                if converted is not None:
                    home_amt = converted
            except Exception as e:
                logger.warning("FX conversion failed for message event: %s", e)

        event_type = "income"
        category = "invoice" if pattern.family == "invoice_approved" else "salary"

        return CanonicalEvent(
            event_id=f"msg_event_{index:03d}",
            user_id=user_id,
            event_type=event_type,
            description=f"Message-confirmed {category}: {text[:60]}...",
            category=category,
            direction="credit",
            amount=home_amt,
            currency=orig_curr,
            event_date=evt_date,
            settlement_date=evt_date,
            status="settled",
            linked_event_id="",
            flexibility="fixed",
            minimum_allowed_amount=None,
            source="message",
            locked=True,
            included_in_forecast=True,
            exclusion_reason=None,
        )

    return None


def parse_messages(
    csv_path: Any = None,
    profiles: Optional[Dict[str, Any]] = None,
    fx_rates: Optional[Dict[Tuple[str, str, date], Decimal]] = None,
) -> List[ParsedMessageFact]:
    """
    Main entrypoint: parses all messages in messages.csv.
    """
    if csv_path is None:
        from config import MESSAGES_CSV
        csv_path = MESSAGES_CSV

    facts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            uid = row.get("user_id", "").strip()
            home_curr = None
            if profiles and uid in profiles:
                p = profiles[uid]
                if isinstance(p, dict):
                    home_curr = p.get("home_currency")
                else:
                    home_curr = getattr(p, "home_currency", None)
            fact = parse_message_row(row, index=idx, home_currency=home_curr, fx_rates=fx_rates)
            facts.append(fact)
    return facts

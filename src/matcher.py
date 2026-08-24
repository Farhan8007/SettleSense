"""Pass 1 — Exact UTR match between bank credits and settlement batches.
Pass 2 — Batch sum matching: match unmatched bank credits to settlement batches
by summing net amounts within a 3-day date window.
Pass 3 — AI-assisted narration parsing for remaining unmatched credits.

The 3-day window rationale:
- Settlement cycles typically take T+1 to T+3 business days
- Bank processing delays (weekends, holidays) can shift dates by 1-2 days
- A 3-day window captures the vast majority of legitimate matches while
  limiting false positives from amount collisions
- This is narrower than the industry-standard T+5 window, reducing ambiguity
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from audit_log import AuditEvent, AuditLog, MatchMethod, MatchStatus
from settlement_math import to_paise


# Tolerance in paise (₹1.00)
TOLERANCE_PAISE = 100

# Maximum date difference in days for batch matching
MAX_DATE_DIFF_DAYS = 3


@dataclass(frozen=True)
class BankCredit:
    bank_credit_id: str
    utr: Optional[str]
    narration: str
    amount_paise: int
    date: str


@dataclass(frozen=True)
class SettlementBatch:
    batch_id: str
    utr: Optional[str]
    net_paise: int
    settlement_date: str
    payment_ids: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    bank_credit_id: str
    batch_id: str
    method: MatchMethod
    amount_paise: int
    payment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExceptionRecord:
    bank_credit_id: str
    exception_type: str
    evidence: dict


def _extract_utr(narration: str, ref_number: str) -> Optional[str]:
    """Extract UTR from narration or ref_number."""
    patterns = [
        r'UTR([A-Z0-9]+)',
        r'NEFT[_-]RAZORPAY[_-]UTR([A-Z0-9]+)',
        r'RTGS[_-]RAZORPAY[_-]UTR([A-Z0-9]+)',
    ]
    for text in (narration, ref_number):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    if ref_number and ref_number.startswith("NEFT"):
        return ref_number[4:].upper()
    if ref_number and ref_number.startswith("RTGS"):
        return ref_number[4:].upper()
    return None


def _normalize_utr(utr: Optional[str]) -> Optional[str]:
    """Normalize UTR by removing known prefixes."""
    if not utr:
        return None
    for prefix in ("UTR", "RTGS", "NEFT"):
        if utr.startswith(prefix):
            return utr[len(prefix):]
    return utr


def _validate_paise(value: int, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be int paise, got bool {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be int paise, got {type(value).__name__} {value!r}")
    return value


def _parse_date(date_str: str) -> datetime.date:
    """Parse YYYY-MM-DD date string."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _date_diff_days(date1: str, date2: str) -> int:
    """Absolute difference in days between two YYYY-MM-DD date strings."""
    return abs((_parse_date(date1) - _parse_date(date2)).days)


def load_bank_credits(path: Path) -> list[BankCredit]:
    credits = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            utr = _extract_utr(row["narration"], row.get("ref_number", ""))
            amount_paise = _validate_paise(to_paise(row["amount"]), "amount_paise")
            credits.append(BankCredit(
                bank_credit_id=f"bank_{i}",
                utr=utr,
                narration=row["narration"],
                amount_paise=amount_paise,
                date=row["date"],
            ))
    return credits


def load_settlement_batches(path: Path) -> list[SettlementBatch]:
    by_batch: dict[str, dict] = defaultdict(lambda: {
        "utr": None,
        "net_paise": 0,
        "settlement_date": "",
        "payment_ids": [],
    })
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_id = row["settlement_id"]
            utr = _normalize_utr(row.get("utr", "").strip().upper() or None)
            net_paise = _validate_paise(to_paise(row["net_amount"]), "net_paise")
            data = by_batch[batch_id]
            if utr and data["utr"] is None:
                data["utr"] = utr
            if not data["settlement_date"]:
                data["settlement_date"] = row["settled_at"]
            data["net_paise"] += net_paise
            data["payment_ids"].append(row["payment_id"])

    batches = []
    for batch_id, data in by_batch.items():
        batches.append(SettlementBatch(
            batch_id=batch_id,
            utr=data["utr"],
            net_paise=data["net_paise"],
            settlement_date=data["settlement_date"][:10],
            payment_ids=tuple(data["payment_ids"]),
        ))
    return batches


def load_settlement_batches_for_pass2(path: Path) -> list[SettlementBatch]:
    """Load settlement batches for Pass 2 — includes all batches, even those with UTR.
    
    Aggregates by settlement_id, sums net_amount, collects payment_ids.
    """
    by_batch: dict[str, dict] = defaultdict(lambda: {
        "utr": None,
        "net_paise": 0,
        "settlement_date": "",
        "payment_ids": [],
    })
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_id = row["settlement_id"]
            utr = _normalize_utr(row.get("utr", "").strip().upper() or None)
            net_paise = _validate_paise(to_paise(row["net_amount"]), "net_paise")
            data = by_batch[batch_id]
            if utr and data["utr"] is None:
                data["utr"] = utr
            if not data["settlement_date"]:
                data["settlement_date"] = row["settled_at"]
            data["net_paise"] += net_paise
            data["payment_ids"].append(row["payment_id"])

    batches = []
    for batch_id, data in by_batch.items():
        batches.append(SettlementBatch(
            batch_id=batch_id,
            utr=data["utr"],
            net_paise=data["net_paise"],
            settlement_date=data["settlement_date"][:10],
            payment_ids=tuple(data["payment_ids"]),
        ))
    return batches


def pass1_exact_utr(
    bank_credits: list[BankCredit],
    settlement_batches: list[SettlementBatch],
    audit_log: AuditLog,
) -> tuple[list[MatchResult], list[BankCredit]]:
    """Match bank credits to settlement batches by exact UTR.

    Returns:
        matched: list of MatchResult for successful matches
        unmatched: list of BankCredit that need further passes
    """
    batch_by_utr: dict[str, SettlementBatch] = {}
    for batch in settlement_batches:
        if batch.utr:
            if batch.utr in batch_by_utr:
                raise ValueError(f"Duplicate UTR in settlement batches: {batch.utr}")
            batch_by_utr[batch.utr] = batch

    utr_counts: dict[str, int] = defaultdict(int)
    for credit in bank_credits:
        if credit.utr:
            utr_counts[credit.utr] += 1

    # Track which UTRs we've already seen (for duplicate detection)
    seen_utrs: set[str] = set()

    matched: list[MatchResult] = []
    unmatched: list[BankCredit] = []

    for credit in bank_credits:
        if not credit.utr:
            unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.PENDING,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason="No UTR extracted from narration",
            ))
            continue

        # Duplicate UTR handling: first occurrence proceeds normally,
        # subsequent ones are flagged as duplicate_credit exceptions
        if credit.utr in seen_utrs:
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"Duplicate UTR in bank statement: {credit.utr}",
                metadata={"exception_type": "duplicate_credit", "utr": credit.utr},
            ))
            # Don't add to unmatched - it's an exception, not pending
            continue

        seen_utrs.add(credit.utr)

        batch = batch_by_utr.get(credit.utr)
        if batch is None:
            unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.PENDING,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"No settlement batch found for UTR: {credit.utr}",
                metadata={"utr": credit.utr},
            ))
            continue

        delta = credit.amount_paise - batch.net_paise
        if delta != 0:
            # Amount mismatch - send to Pass 2 for variance handling
            unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.PENDING,
                match_method=MatchMethod.NONE,
                expected_amount_paise=batch.net_paise,
                actual_amount_paise=credit.amount_paise,
                reason=f"UTR match but amount mismatch (delta={delta} paise), deferring to Pass 2",
                metadata={"utr": credit.utr, "batch_id": batch.batch_id, "delta_paise": delta},
            ))
            continue

        matched.append(MatchResult(
            bank_credit_id=credit.bank_credit_id,
            batch_id=batch.batch_id,
            method=MatchMethod.UTR_EXACT,
            amount_paise=credit.amount_paise,
            payment_ids=batch.payment_ids,
        ))
        audit_log.append(AuditEvent.create(
            bank_record_id=credit.bank_credit_id,
            status=MatchStatus.MATCHED,
            match_method="exact_utr",
            expected_amount_paise=batch.net_paise,
            actual_amount_paise=credit.amount_paise,
            settlement_ids=[batch.batch_id],
            payment_ids=list(batch.payment_ids),
            confidence=1.0,
            reason=f"Exact UTR match: {credit.utr}",
        ))

    return matched, unmatched


def run_pass1(
    bank_path: Path,
    settlement_path: Path,
    audit_log_path: Path,
) -> tuple[list[MatchResult], list[BankCredit]]:
    """Run Pass 1 and write audit log."""
    bank_credits = load_bank_credits(bank_path)
    settlement_batches = load_settlement_batches(settlement_path)
    audit_log = AuditLog(audit_log_path, fsync=False)
    return pass1_exact_utr(bank_credits, settlement_batches, audit_log)


def run_pass2(
    unmatched: list[BankCredit],
    matched_so_far: list[MatchResult],
    settlement_path: Path,
    audit_log_path: Path,
) -> tuple[list[MatchResult], list[BankCredit], list[ExceptionRecord]]:
    """Pass 2 — Batch sum matching within 3-day date window.
    
    Args:
        unmatched: BankCredit objects from Pass 1 that need matching
        matched_so_far: MatchResult objects from Pass 1 (to exclude their batch_ids)
        settlement_path: Path to razorpay_settlements.csv
        audit_log_path: Path for Pass 2 audit log (separate file)
    
    Returns:
        newly_matched: list of MatchResult
        still_unmatched: list of BankCredit (goes to Pass 3)
        exceptions: list of ExceptionRecord
    """
    # Load all settlement batches
    all_batches = load_settlement_batches_for_pass2(settlement_path)
    
    # Exclude batch_ids already matched in Pass 1
    matched_batch_ids = {m.batch_id for m in matched_so_far}
    available_batches = [b for b in all_batches if b.batch_id not in matched_batch_ids]
    
    # Sort unmatched bank credits by date ascending (deterministic order)
    unmatched_sorted = sorted(unmatched, key=lambda c: c.date)
    
    audit_log = AuditLog(audit_log_path, fsync=False)
    
    newly_matched: list[MatchResult] = []
    still_unmatched: list[BankCredit] = []
    exceptions: list[ExceptionRecord] = []
    
    # Track which batches have been consumed (one-to-one constraint)
    consumed_batch_ids = set()
    
    for credit in unmatched_sorted:
        # Find candidates within date window
        candidates = []
        for batch in available_batches:
            if batch.batch_id in consumed_batch_ids:
                continue
            diff = _date_diff_days(credit.date, batch.settlement_date)
            if diff <= MAX_DATE_DIFF_DAYS:
                delta = credit.amount_paise - batch.net_paise
                candidates.append({
                    "batch": batch,
                    "net_paise": batch.net_paise,
                    "delta_paise": delta,
                    "date_diff_days": diff,
                })
        
        # Sort candidates by date_diff (closest first), then by absolute delta
        candidates.sort(key=lambda c: (c["date_diff_days"], abs(c["delta_paise"])))
        
        # Build candidates_considered for audit log
        candidates_considered = [
            {
                "batch_id": c["batch"].batch_id,
                "net_paise": c["net_paise"],
                "delta_paise": c["delta_paise"],
                "date_diff_days": c["date_diff_days"],
            }
            for c in candidates
        ]
        
        # Determine outcome based on candidates
        if not candidates:
            # Case 7: No candidates in window
            still_unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.PENDING,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason="No candidate batch in date window",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "exception_type": "no_candidate_in_window",
                    "flags": {"zero_value": False, "refund": False},
                },
            ))
            continue
        
        # Prioritize UTR-matched batch if bank credit has UTR
        utr_matched_candidates = []
        if credit.utr:
            for c in candidates:
                if c["batch"].utr and c["batch"].utr == credit.utr:
                    utr_matched_candidates.append(c)
        
        # If UTR-matched candidate exists, use it as primary
        primary_candidates = utr_matched_candidates if utr_matched_candidates else candidates
        
        # Check for zero value settlement
        zero_candidates = [c for c in primary_candidates if c["net_paise"] == 0 and credit.amount_paise == 0]
        if zero_candidates:
            # Case 5: Zero value settlement
            chosen = zero_candidates[0]
            consumed_batch_ids.add(chosen["batch"].batch_id)
            newly_matched.append(MatchResult(
                bank_credit_id=credit.bank_credit_id,
                batch_id=chosen["batch"].batch_id,
                method=MatchMethod.BATCH_SUM_EXACT,
                amount_paise=credit.amount_paise,
                payment_ids=chosen["batch"].payment_ids,
            ))
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.MATCHED,
                match_method=MatchMethod.BATCH_SUM_EXACT,
                expected_amount_paise=chosen["net_paise"],
                actual_amount_paise=credit.amount_paise,
                settlement_ids=[chosen["batch"].batch_id],
                payment_ids=list(chosen["batch"].payment_ids),
                confidence=1.0,
                reason=f"Zero-value settlement matched (net=0, amount=0)",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "flags": {"zero_value": True, "refund": False},
                },
            ))
            continue
        
        # Check for refund batch (negative net)
        refund_candidates = [c for c in primary_candidates if c["net_paise"] < 0 and credit.amount_paise < 0]
        if refund_candidates:
            exact_refund = [c for c in refund_candidates if c["delta_paise"] == 0]
            if len(exact_refund) == 1:
                # Case 6: Refund batch with exact match
                chosen = exact_refund[0]
                consumed_batch_ids.add(chosen["batch"].batch_id)
                newly_matched.append(MatchResult(
                    bank_credit_id=credit.bank_credit_id,
                    batch_id=chosen["batch"].batch_id,
                    method=MatchMethod.BATCH_SUM_EXACT,
                    amount_paise=credit.amount_paise,
                    payment_ids=chosen["batch"].payment_ids,
                ))
                audit_log.append(AuditEvent.create(
                    bank_record_id=credit.bank_credit_id,
                    status=MatchStatus.MATCHED,
                    match_method=MatchMethod.BATCH_SUM_EXACT,
                    expected_amount_paise=chosen["net_paise"],
                    actual_amount_paise=credit.amount_paise,
                    settlement_ids=[chosen["batch"].batch_id],
                    payment_ids=list(chosen["batch"].payment_ids),
                    confidence=1.0,
                    reason=f"Refund batch matched exactly",
                    metadata={
                        "pass": 2,
                        "candidates_considered": candidates_considered,
                        "flags": {"zero_value": False, "refund": True},
                    },
                ))
                continue
        
        # Filter to exact delta matches from primary candidates
        exact_matches = [c for c in primary_candidates if c["delta_paise"] == 0]
        
        if len(exact_matches) == 1:
            # Case 1: Exactly one candidate, delta == 0
            chosen = exact_matches[0]
            consumed_batch_ids.add(chosen["batch"].batch_id)
            newly_matched.append(MatchResult(
                bank_credit_id=credit.bank_credit_id,
                batch_id=chosen["batch"].batch_id,
                method=MatchMethod.BATCH_SUM_EXACT,
                amount_paise=credit.amount_paise,
                payment_ids=chosen["batch"].payment_ids,
            ))
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.MATCHED,
                match_method=MatchMethod.BATCH_SUM_EXACT,
                expected_amount_paise=chosen["net_paise"],
                actual_amount_paise=credit.amount_paise,
                settlement_ids=[chosen["batch"].batch_id],
                payment_ids=list(chosen["batch"].payment_ids),
                confidence=1.0,
                reason=f"Batch sum exact match within {MAX_DATE_DIFF_DAYS}-day window",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "flags": {"zero_value": False, "refund": False},
                },
            ))
            continue
        
        if len(exact_matches) > 1:
            # Case 4: Multiple candidates with delta == 0
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type="ambiguous_amount",
                evidence={
                    "candidates": candidates_considered,
                    "matching_batch_ids": [c["batch"].batch_id for c in exact_matches],
                },
            )
            exceptions.append(exception)
            still_unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"Ambiguous: {len(exact_matches)} batches match exactly within window",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "exception_type": "ambiguous_amount",
                    "flags": {"zero_value": False, "refund": False},
                },
            ))
            continue
        
        # Check tolerance matches (0 < |delta| <= 100) from primary candidates
        tolerance_matches = [c for c in primary_candidates if 0 < abs(c["delta_paise"]) <= TOLERANCE_PAISE]
        if len(tolerance_matches) == 1 and len(primary_candidates) == 1:
            # Case 2: Exactly one candidate, within tolerance
            chosen = tolerance_matches[0]
            consumed_batch_ids.add(chosen["batch"].batch_id)
            newly_matched.append(MatchResult(
                bank_credit_id=credit.bank_credit_id,
                batch_id=chosen["batch"].batch_id,
                method=MatchMethod.BATCH_SUM_TOLERANCE,
                amount_paise=credit.amount_paise,
                payment_ids=chosen["batch"].payment_ids,
            ))
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.MATCHED,
                match_method=MatchMethod.BATCH_SUM_TOLERANCE,
                expected_amount_paise=chosen["net_paise"],
                actual_amount_paise=credit.amount_paise,
                settlement_ids=[chosen["batch"].batch_id],
                payment_ids=list(chosen["batch"].payment_ids),
                confidence=0.9,
                reason=f"Batch sum within tolerance ({abs(chosen['delta_paise'])} paise)",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "flags": {"zero_value": False, "refund": False},
                },
            ))
            continue
        
        # Check if exactly one primary candidate but delta > tolerance
        if len(primary_candidates) == 1 and abs(primary_candidates[0]["delta_paise"]) > TOLERANCE_PAISE:
            # Case 3: Exactly one candidate, delta > 100 paise
            chosen = primary_candidates[0]
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type="variance_breach",
                evidence={
                    "candidate": candidates_considered[0],
                    "delta_paise": chosen["delta_paise"],
                    "tolerance_paise": TOLERANCE_PAISE,
                },
            )
            exceptions.append(exception)
            still_unmatched.append(credit)
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=chosen["net_paise"],
                actual_amount_paise=credit.amount_paise,
                reason=f"Variance breach: delta {abs(chosen['delta_paise'])} paise exceeds tolerance {TOLERANCE_PAISE}",
                metadata={
                    "pass": 2,
                    "candidates_considered": candidates_considered,
                    "exception_type": "variance_breach",
                    "flags": {"zero_value": False, "refund": False},
                },
            ))
            continue
        
        # Default: no clear match, send to Pass 3
        still_unmatched.append(credit)
        audit_log.append(AuditEvent.create(
            bank_record_id=credit.bank_credit_id,
            status=MatchStatus.PENDING,
            match_method=MatchMethod.NONE,
            expected_amount_paise=0,
            actual_amount_paise=credit.amount_paise,
            reason="No unique match found in window",
            metadata={
                "pass": 2,
                "candidates_considered": candidates_considered,
                "exception_type": "no_unique_match",
                "flags": {"zero_value": False, "refund": False},
            },
        ))
    
    return newly_matched, still_unmatched, exceptions


# ============================================================================
# Pass 3 — AI-assisted narration parsing
# ============================================================================

MAX_GROQ_CALLS = 20
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 10  # seconds


def _build_batch_index(batches: list[SettlementBatch]) -> dict[str, SettlementBatch]:
    """Build a lookup index by UTR from settlement batches."""
    index: dict[str, SettlementBatch] = {}
    for batch in batches:
        if batch.utr:
            index[batch.utr] = batch
    return index


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_groq(narration: str, api_key: str) -> str:
    """Call Groq API to extract payment reference from narration.
    
    Returns the raw response text from the API.
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")
    
    client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT)
    
    system_prompt = (
        "You are a bank statement parser for Indian payments. "
        "Extract any payment reference from the narration text. "
        "Reply with JSON only. No explanation. No markdown. "
        'Schema: {"extracted_ref": "string or null", "confidence": "high or low", '
        '"ref_type": "utr or payment_id or merchant_ref or unknown"}'
    )
    user_prompt = f"Narration: {narration}"
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=100,
        temperature=0,
    )
    
    return response.choices[0].message.content


def _parse_ai_response(raw: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse AI response JSON.
    
    Returns: (extracted_ref, confidence, ref_type)
    If JSON is invalid or extracted_ref is null, returns (None, None, None)
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None, None
    
    extracted_ref = data.get("extracted_ref")
    confidence = data.get("confidence")
    ref_type = data.get("ref_type")
    
    if extracted_ref is None:
        return None, confidence, ref_type
    if not isinstance(extracted_ref, str) or not extracted_ref.strip():
        return None, confidence, ref_type
    
    return extracted_ref.strip(), confidence, ref_type


def run_pass3(
    still_unmatched: list[BankCredit],
    batch_index: list[SettlementBatch],
    matched_so_far: list[MatchResult],
) -> tuple[list[MatchResult], list[ExceptionRecord]]:
    """Pass 3 — AI-assisted narration parsing for remaining unmatched credits.
    
    Args:
        still_unmatched: BankCredit objects from Pass 2 that need matching
        batch_index: SettlementBatch objects (aggregated by settlement_id) from Pass 2
        matched_so_far: All MatchResult objects from Pass 1 and Pass 2 (to exclude their bank_credit_ids and batch_ids)
    
    Returns:
        newly_matched: list of MatchResult (method="ai_assisted_utr")
        exceptions: list of ExceptionRecord
    """
    # Build UTR lookup from batch_index
    utr_to_batch = _build_batch_index(batch_index)
    
    # Track already matched bank_credit_ids and batch_ids
    matched_bank_ids = {m.bank_credit_id for m in matched_so_far}
    consumed_batch_ids = {m.batch_id for m in matched_so_far}
    
    # Check if we need Groq API key (only if any record has utr=None)
    needs_ai = any(c.utr is None for c in still_unmatched if c.bank_credit_id not in matched_bank_ids)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    groq_api_key = os.environ.get("GROQ_API_KEY")
    fallback_mode = False
    
    if needs_ai and not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set in environment. "
            "Get a free key at https://console.groq.com and export GROQ_API_KEY=your_key"
        )
    
    audit_log = AuditLog("out/audit_pass3.jsonl", fsync=False)
    
    newly_matched: list[MatchResult] = []
    exceptions: list[ExceptionRecord] = []
    groq_call_count = 0
    
    for credit in still_unmatched:
        # Skip if already matched in earlier passes
        if credit.bank_credit_id in matched_bank_ids:
            continue
        
        # Initialize audit fields
        ai_called = False
        ai_raw_response = None
        extracted_ref = None
        confidence = None
        ref_type = None
        retry_lookup_attempted = False
        retry_lookup_found = False
        status = "exception"
        match_method = None
        matched_batch_id = None
        exception_type = None
        fallback_mode_record = False
        
        # Step A — Pre-check: UTR exists but wasn't matched in Pass 1
        if credit.utr is not None:
            # UTR exists but no batch matched in Pass 1 (amount mismatch or no batch)
            exception_type = "utr_found_no_matching_batch"
            evidence = {"utr": credit.utr}
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type=exception_type,
                evidence=evidence,
            )
            exceptions.append(exception)
            status = "exception"
            
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"UTR found but no matching batch: {credit.utr}",
                metadata={
                    "pass": 3,
                    "narration": credit.narration,
                    "utr_before_ai": credit.utr,
                    "ai_called": False,
                    "ai_raw_response": None,
                    "extracted_ref": None,
                    "confidence": None,
                    "ref_type": None,
                    "retry_lookup_attempted": False,
                    "retry_lookup_found": False,
                    "exception_type": exception_type,
                    "fallback_mode": False,
                },
            ))
            continue
        
        # Step B — AI narration parse (only for utr=None records)
        if groq_call_count >= MAX_GROQ_CALLS:
            raise RuntimeError(f"Groq API call limit ({MAX_GROQ_CALLS}) exceeded")
        
        try:
            ai_raw_response = _call_groq(credit.narration, groq_api_key)
            ai_called = True
            groq_call_count += 1
        except Exception as e:
            # Step E — Fallback: Groq unavailable
            fallback_mode = True
            fallback_mode_record = True
            print(f"⚠ Groq unavailable — Pass 3 running in fallback mode: {e}")
            
            # Treat as unresolvable
            exception_type = "ai_service_unavailable"
            evidence = {"error": str(e), "narration": credit.narration}
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type=exception_type,
                evidence=evidence,
            )
            exceptions.append(exception)
            status = "exception"
            
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"Groq API unavailable: {e}",
                metadata={
                    "pass": 3,
                    "narration": credit.narration,
                    "utr_before_ai": None,
                    "ai_called": False,
                    "ai_raw_response": None,
                    "extracted_ref": None,
                    "confidence": None,
                    "ref_type": None,
                    "retry_lookup_attempted": False,
                    "retry_lookup_found": False,
                    "exception_type": exception_type,
                    "fallback_mode": True,
                },
            ))
            continue
        
        # Step C — Validate LLM response
        extracted_ref, confidence, ref_type = _parse_ai_response(ai_raw_response)
        
        if extracted_ref is None or confidence != "high":
            # Treat as unresolvable
            exception_type = "unparseable_narration_unresolvable"
            evidence = {
                "narration": credit.narration,
                "ai_raw_response": ai_raw_response,
                "extracted_ref": extracted_ref,
                "confidence": confidence,
                "ref_type": ref_type,
            }
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type=exception_type,
                evidence=evidence,
            )
            exceptions.append(exception)
            status = "exception"
            
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason="AI could not extract a reliable reference from narration",
                metadata={
                    "pass": 3,
                    "narration": credit.narration,
                    "utr_before_ai": None,
                    "ai_called": ai_called,
                    "ai_raw_response": ai_raw_response,
                    "extracted_ref": extracted_ref,
                    "confidence": confidence,
                    "ref_type": ref_type,
                    "retry_lookup_attempted": False,
                    "retry_lookup_found": False,
                    "exception_type": exception_type,
                    "fallback_mode": fallback_mode_record,
                },
            ))
            continue
        
        # Step D — Retry lookup with extracted reference
        retry_lookup_attempted = True
        batch = utr_to_batch.get(extracted_ref)
        
        if batch is not None and batch.batch_id not in consumed_batch_ids:
            retry_lookup_found = True
            consumed_batch_ids.add(batch.batch_id)
            
            match_result = MatchResult(
                bank_credit_id=credit.bank_credit_id,
                batch_id=batch.batch_id,
                method=MatchMethod.AI_ASSISTED_UTR,
                amount_paise=credit.amount_paise,
                payment_ids=batch.payment_ids,
            )
            newly_matched.append(match_result)
            status = "matched"
            match_method = "ai_assisted_utr"
            matched_batch_id = batch.batch_id
            
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.MATCHED,
                match_method=MatchMethod.AI_ASSISTED_UTR,
                expected_amount_paise=batch.net_paise,
                actual_amount_paise=credit.amount_paise,
                settlement_ids=[batch.batch_id],
                payment_ids=list(batch.payment_ids),
                confidence=1.0,
                reason=f"AI-assisted UTR match: extracted '{extracted_ref}' from narration",
                metadata={
                    "pass": 3,
                    "narration": credit.narration,
                    "utr_before_ai": None,
                    "ai_called": ai_called,
                    "ai_raw_response": ai_raw_response,
                    "extracted_ref": extracted_ref,
                    "confidence": confidence,
                    "ref_type": ref_type,
                    "retry_lookup_attempted": True,
                    "retry_lookup_found": True,
                    "matched_batch_id": batch.batch_id,
                    "fallback_mode": fallback_mode_record,
                },
            ))
        else:
            # No match found for extracted reference
            exception_type = "unparseable_narration_unresolvable"
            evidence = {
                "narration": credit.narration,
                "ai_raw_response": ai_raw_response,
                "extracted_ref": extracted_ref,
                "confidence": confidence,
                "ref_type": ref_type,
                "batch_found": batch is not None,
                "batch_already_consumed": batch is not None and batch.batch_id in consumed_batch_ids,
            }
            exception = ExceptionRecord(
                bank_credit_id=credit.bank_credit_id,
                exception_type=exception_type,
                evidence=evidence,
            )
            exceptions.append(exception)
            status = "exception"
            
            audit_log.append(AuditEvent.create(
                bank_record_id=credit.bank_credit_id,
                status=MatchStatus.EXCEPTION,
                match_method=MatchMethod.NONE,
                expected_amount_paise=0,
                actual_amount_paise=credit.amount_paise,
                reason=f"Extracted reference '{extracted_ref}' not found in settlement batches",
                metadata={
                    "pass": 3,
                    "narration": credit.narration,
                    "utr_before_ai": None,
                    "ai_called": ai_called,
                    "ai_raw_response": ai_raw_response,
                    "extracted_ref": extracted_ref,
                    "confidence": confidence,
                    "ref_type": ref_type,
                    "retry_lookup_attempted": True,
                    "retry_lookup_found": False,
                    "exception_type": exception_type,
                    "fallback_mode": fallback_mode_record,
                },
            ))
    
    # Demo hook print
    print(f"Pass 3 complete: {len(newly_matched)} AI-assisted matches, {len(exceptions)} unresolvable → exception queue")
    
    return newly_matched, exceptions


if __name__ == "__main__":
    data_dir = Path("data")
    audit_dir = Path("out")
    audit_dir.mkdir(exist_ok=True)

    matched, unmatched = run_pass1(
        data_dir / "bank_statement.csv",
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass1.jsonl",
    )

    print(f"Pass 1 complete:")
    print(f"  Matched: {len(matched)}")
    print(f"  Unmatched (for next pass): {len(unmatched)}")
    for m in matched:
        print(f"  {m.bank_credit_id} -> {m.batch_id} ({m.amount_paise} paise)")

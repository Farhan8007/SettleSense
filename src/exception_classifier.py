"""Exception Classifier for SettleSense.

Takes the complete exception picture from all three matching passes, enriches
each existing bank-side exception with structured evidence, and classifies the
settlement-side exceptions (unmatched batches and unsettled orders) that the
matching passes never touched. Produces a final ExceptionReport.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from audit_log import AuditLog
from matcher import BankCredit, ExceptionRecord, MatchResult, load_bank_credits
from settlement_math import fmt, to_paise


# Precedence for deduplicating bank credits seen in more than one pass.
# A bank credit keeps the most specific earlier-pass classification.
_TYPE_PRIORITY = {
    "duplicate_credit": 5,
    "variance_breach": 4,
    "ambiguous_amount": 3,
    "utr_found_no_matching_batch": 2,
    "ai_service_unavailable": 1,
}


@dataclass(frozen=True)
class ClassifiedException:
    bank_credit_id: Optional[str]
    batch_id: Optional[str]
    order_id: Optional[str]
    exception_type: str
    evidence: dict
    severity: str
    requires_human_review: bool


@dataclass(frozen=True)
class ExceptionReport:
    total_exceptions: int
    by_type: dict
    by_severity: dict
    requires_human_review_count: int
    exceptions: list = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> "datetime.date":
    return datetime.now(timezone.utc).date()


def _days_since(date_str: str) -> int:
    """Difference in days between a YYYY-MM-DD[ HH:MM:SS] and today."""
    d = date_str[:10]
    date = datetime.strptime(d, "%Y-%m-%d").date()
    return (_today() - date).days


def _as_int(value, field_name: str) -> int:
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{field_name} must be int paise, got {type(value).__name__} {value!r}"
    )
    return value


def _load_settlement_batches(path: Path) -> dict[str, dict]:
    """Load settlements CSV, aggregated per batch.

    Returns a dict batch_id -> {
        "net_paise", "gross_paise", "payment_ids", "settled_at", "utr"
    }
    """
    by_batch: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_id = row["settlement_id"]
            net_paise = to_paise(row["net_amount"])
            gross_paise = to_paise(row["amount"])
            utr_val = (row.get("utr") or "").strip() or None
            if batch_id not in by_batch:
                by_batch[batch_id] = {
                    "net_paise": 0,
                    "gross_paise": 0,
                    "payment_ids": [],
                    "settled_at": row["settled_at"],
                    "utr": utr_val,
                }
            data = by_batch[batch_id]
            data["net_paise"] += net_paise
            data["gross_paise"] += gross_paise
            data["payment_ids"].append(row["payment_id"])
            if utr_val and data["utr"] is None:
                data["utr"] = utr_val
    return by_batch


def _load_orders(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _settled_payment_ids(settlements: dict[str, dict]) -> set[str]:
    ids = set()
    for data in settlements.values():
        ids.update(data["payment_ids"])
    return ids


def _idx(bank: BankCredit) -> int:
    try:
        return int(bank.bank_credit_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _first_bank_for_utr(bank_by_id: dict[str, BankCredit], utr: str) -> Optional[BankCredit]:
    """Return the earliest bank credit record carrying the given UTR."""
    matches = [b for b in bank_by_id.values() if b.utr == utr]
    if not matches:
        return None
    return min(matches, key=_idx)


# ---------------------------------------------------------------------------
# Enrichment per exception type
# ---------------------------------------------------------------------------

def _enrich(exc: ExceptionRecord, ctx: dict) -> dict:
    etype = exc.exception_type
    raw = exc.evidence or {}
    bank_by_id = ctx["bank_by_id"]
    settlements = ctx["settlements"]
    bank = bank_by_id.get(exc.bank_credit_id)

    if etype == "duplicate_credit":
        utr = raw.get("utr") or (bank.utr if bank else None)
        first = None
        if utr:
            fb = _first_bank_for_utr(bank_by_id, utr)
            if fb:
                first = fb.bank_credit_id
        amount = raw.get("amount_paise")
        if amount is None and bank:
            amount = bank.amount_paise
        return {
            "utr": utr,
            "first_seen_in": first,
            "amount_paise": _as_int(amount, "amount_paise"),
        }

    if etype == "variance_breach":
        candidate = raw.get("candidate") or {}
        candidate_batch_id = candidate.get("batch_id") or raw.get("candidate_batch_id")
        batch_net_paise = candidate.get("net_paise")
        if batch_net_paise is None:
            batch_net_paise = raw.get("batch_net_paise")
        delta_paise = raw.get("delta_paise")
        if delta_paise is None:
            delta_paise = candidate.get("delta_paise")

        if candidate_batch_id and candidate_batch_id in settlements:
            binfo = settlements[candidate_batch_id]
            if batch_net_paise is None:
                batch_net_paise = binfo["net_paise"]

        bank_amount_paise = raw.get("bank_amount_paise")
        if bank_amount_paise is None and delta_paise is not None and batch_net_paise is not None:
            bank_amount_paise = batch_net_paise + delta_paise

        delta_paise = _as_int(delta_paise, "delta_paise") if delta_paise is not None else 0
        bank_amount_paise = _as_int(bank_amount_paise, "bank_amount_paise")
        if batch_net_paise is not None:
            batch_net_paise = _as_int(batch_net_paise, "batch_net_paise")

        gross = None
        if candidate_batch_id and candidate_batch_id in settlements:
            gross = settlements[candidate_batch_id]["gross_paise"]

        if gross is not None and gross == bank_amount_paise:
            cause = "fees_not_deducted"
        elif delta_paise > 0:
            cause = "partial_payment"
        elif delta_paise < 0:
            cause = "extra_deduction"
        else:
            cause = "unknown"

        return {
            "candidate_batch_id": candidate_batch_id,
            "bank_amount_paise": bank_amount_paise,
            "batch_net_paise": batch_net_paise,
            "delta_paise": delta_paise,
            "delta_rupees": fmt(delta_paise),
            "likely_cause": cause,
        }

    if etype == "ambiguous_amount":
        candidate_ids = raw.get("matching_batch_ids") or [
            c["batch_id"] for c in raw.get("candidates", [])
        ]
        bank_amount = raw.get("bank_amount_paise")
        if bank_amount is None and raw.get("candidates"):
            for c in raw["candidates"]:
                if c.get("delta_paise") == 0:
                    bank_amount = c.get("net_paise")
                    break
        if bank_amount is None and bank:
            bank_amount = bank.amount_paise
        return {
            "bank_amount_paise": _as_int(bank_amount, "bank_amount_paise"),
            "candidate_batch_ids": list(candidate_ids),
            "candidate_count": _as_int(len(candidate_ids), "candidate_count"),
            "resolution": "manual_review_required",
        }

    if etype == "utr_found_no_matching_batch":
        utr = raw.get("utr") or (bank.utr if bank else None)
        bank_amount = raw.get("bank_amount_paise")
        if bank_amount is None and bank:
            bank_amount = bank.amount_paise
        closest_batch_id = None
        closest_delta = None
        if bank_amount is not None:
            best = None
            for bid, binfo in settlements.items():
                diff = bank_amount - binfo["net_paise"]
                if best is None or abs(diff) < abs(best):
                    best = diff
                    closest_batch_id = bid
            if best is not None:
                closest_delta = best
        return {
            "utr": utr,
            "bank_amount_paise": _as_int(bank_amount, "bank_amount_paise"),
            "searched_batch_count": _as_int(len(settlements), "searched_batch_count"),
            "closest_batch_id": closest_batch_id,
            "closest_delta_paise": closest_delta,
        }

    if etype == "ai_service_unavailable":
        narration = raw.get("narration")
        if narration is None and bank:
            narration = bank.narration
        return {
            "narration": narration,
            "fallback_reason": "groq_unavailable",
            "recommended_action": "retry_with_ai_when_available",
        }

    # Unknown type — return whatever raw evidence exists (never empty).
    return dict(raw) or {"reason": etype}


# ---------------------------------------------------------------------------
# Severity + human review rules
# ---------------------------------------------------------------------------

def _severity(etype: str, evidence: dict) -> str:
    if etype == "variance_breach":
        return "high" if abs(evidence.get("delta_paise", 0)) > 10000 else "medium"
    if etype == "missing_settlement":
        return "high" if evidence.get("days_since_settlement", 0) > 3 else "medium"
    if etype == "duplicate_credit":
        return "high"
    if etype == "ambiguous_amount":
        return "medium"
    if etype == "utr_found_no_matching_batch":
        return "medium"
    if etype == "order_never_settled":
        return "medium" if evidence.get("days_outstanding", 0) > 7 else "low"
    if etype == "ai_service_unavailable":
        return "low"
    if etype == "zero_value_settlement":
        return "low"
    return "low"


_HUMAN_REVIEW_TYPES = {
    "duplicate_credit",
    "variance_breach",
    "ambiguous_amount",
    "missing_settlement",
    "utr_found_no_matching_batch",
}


def _requires_human_review(etype: str) -> bool:
    return etype in _HUMAN_REVIEW_TYPES


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_exceptions(
    pass1_exceptions: list[ExceptionRecord],
    pass2_exceptions: list[ExceptionRecord],
    pass3_exceptions: list[ExceptionRecord],
    all_matched: list[MatchResult],
    settlement_path: str = "data/razorpay_settlements.csv",
    order_path: str = "data/order_ledger.csv",
    audit_path: str = "out/audit_exceptions.jsonl",
) -> ExceptionReport:
    settlements = _load_settlement_batches(Path(settlement_path))
    orders = _load_orders(Path(order_path))
    bank_by_id = {b.bank_credit_id: b for b in load_bank_credits(Path("data/bank_statement.csv"))}

    # -- JOB 1: dedupe + enrich existing bank-side exceptions -----------------
    combined: dict[str, ExceptionRecord] = {}
    for exc in list(pass1_exceptions) + list(pass2_exceptions) + list(pass3_exceptions):
        etype = exc.exception_type
        prev = combined.get(exc.bank_credit_id)
        if prev is None or _TYPE_PRIORITY.get(etype, 0) >= _TYPE_PRIORITY.get(prev.exception_type, 0):
            combined[exc.bank_credit_id] = exc

    ctx = {"bank_by_id": bank_by_id, "settlements": settlements}

    classified: list[ClassifiedException] = []

    for exc in combined.values():
        ev = _enrich(exc, ctx)
        sev = _severity(exc.exception_type, ev)
        review = _requires_human_review(exc.exception_type)
        classified.append(ClassifiedException(
            bank_credit_id=exc.bank_credit_id,
            batch_id=None,
            order_id=None,
            exception_type=exc.exception_type,
            evidence=ev,
            severity=sev,
            requires_human_review=review,
        ))

    # -- JOB 2: classify settlement-side exceptions --------------------------
    matched_batch_ids = {m.batch_id for m in all_matched}
    referenced_batch_ids = set()
    for exc in combined.values():
        ev = exc.evidence or {}
        cand = ev.get("candidate") or {}
        if cand.get("batch_id"):
            referenced_batch_ids.add(cand["batch_id"])
        for c in ev.get("candidates", []):
            if c.get("batch_id"):
                referenced_batch_ids.add(c["batch_id"])
        referenced_batch_ids.update(ev.get("matching_batch_ids", []))

    matched_or_flagged = matched_batch_ids | referenced_batch_ids

    for batch_id in settlements:
        if batch_id in matched_or_flagged:
            continue
        binfo = settlements[batch_id]
        if binfo["gross_paise"] == 0 and binfo["net_paise"] == 0:
            ev = {
                "batch_id": batch_id,
                "payment_ids": list(binfo["payment_ids"]),
                "likely_cause": "full_refund_or_void",
            }
            classified.append(ClassifiedException(
                bank_credit_id=None,
                batch_id=batch_id,
                order_id=None,
                exception_type="zero_value_settlement",
                evidence=ev,
                severity=_severity("zero_value_settlement", ev),
                requires_human_review=_requires_human_review("zero_value_settlement"),
            ))
        else:
            days = _days_since(binfo["settled_at"])
            ev = {
                "batch_id": batch_id,
                "batch_net_paise": _as_int(binfo["net_paise"], "batch_net_paise"),
                "payment_ids": list(binfo["payment_ids"]),
                "settled_at": binfo["settled_at"],
                "days_since_settlement": _as_int(days, "days_since_settlement"),
                "recommended_action": "check_bank_statement_for_date_range",
            }
            classified.append(ClassifiedException(
                bank_credit_id=None,
                batch_id=batch_id,
                order_id=None,
                exception_type="missing_settlement",
                evidence=ev,
                severity=_severity("missing_settlement", ev),
                requires_human_review=_requires_human_review("missing_settlement"),
            ))

    settled_payment_ids = _settled_payment_ids(settlements)
    for order in orders:
        if order["payment_id"] in settled_payment_ids:
            continue
        amount_paise = to_paise(order["amount"])
        days_out = _days_since(order["created_at"])
        ev = {
            "order_id": order["order_id"],
            "amount_paise": _as_int(amount_paise, "amount_paise"),
            "status": order["status"],
            "created_at": order["created_at"],
            "product_category": order["product_category"],
            "days_outstanding": _as_int(days_out, "days_outstanding"),
        }
        classified.append(ClassifiedException(
            bank_credit_id=None,
            batch_id=None,
            order_id=order["order_id"],
            exception_type="order_never_settled",
            evidence=ev,
            severity=_severity("order_never_settled", ev),
            requires_human_review=_requires_human_review("order_never_settled"),
        ))

    # -- audit + summary -----------------------------------------------------
    os.makedirs(Path(audit_path).parent, exist_ok=True)
    with open(audit_path, "a", encoding="utf-8", newline="\n") as f:
        for exc in classified:
            assert isinstance(exc.evidence, dict) and exc.evidence, "evidence must be non-empty"
            assert exc.severity in ("high", "medium", "low"), f"bad severity {exc.severity}"
            assert isinstance(exc.requires_human_review, bool), "requires_human_review must be bool"
            event = {
                "classifier": "exception_classifier",
                "bank_credit_id": exc.bank_credit_id,
                "batch_id": exc.batch_id,
                "order_id": exc.order_id,
                "exception_type": exc.exception_type,
                "evidence": exc.evidence,
                "severity": exc.severity,
                "requires_human_review": exc.requires_human_review,
                "timestamp": _utc_now_iso(),
            }
            f.write(json.dumps(event, ensure_ascii=True) + "\n")

    by_type: dict[str, int] = {}
    by_severity = {"high": 0, "medium": 0, "low": 0}
    review_count = 0
    for ce in classified:
        by_type[ce.exception_type] = by_type.get(ce.exception_type, 0) + 1
        by_severity[ce.severity] += 1
        if ce.requires_human_review:
            review_count += 1

    report = ExceptionReport(
        total_exceptions=len(classified),
        by_type=by_type,
        by_severity=by_severity,
        requires_human_review_count=review_count,
        exceptions=classified,
    )

    print("Exception classifier complete:")
    print(f"  {by_severity['high']} high severity, {by_severity['medium']} medium severity, {by_severity['low']} low severity")
    print(f"  {review_count} require human review")
    return report
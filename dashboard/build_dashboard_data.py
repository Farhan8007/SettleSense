#!/usr/bin/env python3
"""Step 7a — Dashboard data aggregation layer.

Runs ONE fresh, complete pipeline execution (Pass1 → Pass2 → Pass3 →
classify_exceptions) and assembles a single JSON object for the dashboard.
Never reads pre-existing out/*.jsonl files — always starts clean.

Usage:
    python3 dashboard/build_dashboard_data.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow imports from src/ regardless of where we are invoked from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from matcher import (
    run_pass1,
    run_pass2,
    run_pass3,
    load_settlement_batches_for_pass2,
    ExceptionRecord,
    MatchResult,
)
from exception_classifier import classify_exceptions
from razorpay_connector import fetch_test_payments, normalize_razorpay_payment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
AUDIT_DIR = PROJECT_ROOT / "out"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
OUTPUT_PATH = DASHBOARD_DIR / "dashboard_data.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pass1_exceptions(audit_path: Path) -> list:
    """Reconstruct Pass 1 exceptions from audit log (duplicate_credit entries)."""
    excs = []
    if not audit_path.exists():
        return excs
    with audit_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("status") != "exception":
                continue
            etype = e.get("metadata", {}).get("exception_type", "unknown")
            excs.append(ExceptionRecord(
                bank_credit_id=e["bank_record_id"],
                exception_type=etype,
                evidence=e.get("metadata", {}),
            ))
    return excs


def _read_all_audit_events(*audit_paths) -> list:
    """Read all JSONL audit events from the given files in order."""
    events = []
    for path in audit_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    # Sort by seq so the audit trail is in deterministic append order
    events.sort(key=lambda e: e.get("seq", 0))
    return events


def _determine_pass(event: dict):
    """Derive which pass an audit event belongs to."""
    meta = event.get("metadata") or {}
    if "pass" in meta:
        return meta["pass"]
    # Pass 1 events have no 'pass' in metadata
    return 1


# ---------------------------------------------------------------------------
# Step 1 — Clean out/ directory (delete stale JSONL files)
# ---------------------------------------------------------------------------

def clean_output_dir() -> None:
    AUDIT_DIR.mkdir(exist_ok=True)
    removed = []
    for f in AUDIT_DIR.glob("*.jsonl"):
        f.unlink()
        removed.append(f.name)
    for f in AUDIT_DIR.glob("*.jsonl.lock"):
        f.unlink()
    if removed:
        print(f"Cleaned {len(removed)} stale audit file(s): {', '.join(removed)}")
    else:
        print("No stale audit files to clean.")


# ---------------------------------------------------------------------------
# Step 2 — Run full pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    print("Running Pass 1 (exact UTR)...")
    matched_p1, unmatched_p1 = run_pass1(
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "razorpay_settlements.csv",
        AUDIT_DIR / "audit_pass1.jsonl",
    )

    print("Running Pass 2 (batch sum)...")
    newly_matched_p2, still_unmatched_p2, exceptions_p2 = run_pass2(
        unmatched_p1,
        matched_p1,
        DATA_DIR / "razorpay_settlements.csv",
        AUDIT_DIR / "audit_pass2.jsonl",
    )

    print("Running Pass 3 (AI-assisted narration)...")
    all_batches = load_settlement_batches_for_pass2(DATA_DIR / "razorpay_settlements.csv")
    matched_batch_ids = (
        {m.batch_id for m in matched_p1} | {m.batch_id for m in newly_matched_p2}
    )
    batch_index = [b for b in all_batches if b.batch_id not in matched_batch_ids]
    newly_matched_p3, exceptions_p3 = run_pass3(
        still_unmatched_p2,
        batch_index,
        matched_p1 + newly_matched_p2,
    )

    print("Running Exception Classifier...")
    exceptions_p1 = _load_pass1_exceptions(AUDIT_DIR / "audit_pass1.jsonl")
    all_matched = matched_p1 + newly_matched_p2 + newly_matched_p3
    report = classify_exceptions(
        exceptions_p1,
        exceptions_p2,
        exceptions_p3,
        all_matched,
        str(DATA_DIR / "razorpay_settlements.csv"),
        str(DATA_DIR / "order_ledger.csv"),
        str(AUDIT_DIR / "audit_exceptions.jsonl"),
    )

    # Unified exception list used for accuracy comparison
    all_exceptions = (
        exceptions_p1
        + exceptions_p2
        + exceptions_p3
        + report.exceptions
    )

    return all_matched, all_exceptions, report


# ---------------------------------------------------------------------------
# Step 3 — Compare against ground truth (mirrors accuracy_report.py exactly)
# ---------------------------------------------------------------------------

def compare_ground_truth(all_matched, all_exceptions, report):
    """Return (summary_metrics, records_list).

    Replicates the same logic as tests/accuracy_report.py so the dashboard
    cannot diverge from the official accuracy numbers.
    """
    matched_by_id = {m.bank_credit_id: m for m in all_matched}
    exceptions_by_id = {e.bank_credit_id: e for e in all_exceptions}

    # Build lookup for settlement-side and order-side classifier results
    classifier_by_batch = {}
    classifier_by_order = {}
    for exc in report.exceptions:
        if exc.batch_id:
            classifier_by_batch[exc.batch_id] = exc
        if exc.order_id:
            classifier_by_order[exc.order_id] = exc

    # Load ground truth
    with (DATA_DIR / "ground_truth.csv").open() as f:
        gt_rows = list(csv.DictReader(f))

    total_records = 0
    correct_decisions = 0
    false_positives = 0
    false_negatives = 0

    # Deterministic vs AI-assisted method sets
    deterministic_methods = {
        "exact_utr", "utr_exact", "batch_sum_exact",
        "batch_sum_tolerance", "batch_by_utr",
    }
    ai_methods = {"ai_assisted_utr"}

    records = []

    for row in gt_rows:
        record_type = row["record_type"]
        record_id = row["record_id"]
        expected_outcome = row["expected_outcome"]

        total_records += 1
        is_correct = False

        if record_type == "bank_credit":
            actual_matched = record_id in matched_by_id
            actual_exception = record_id in exceptions_by_id
            actual_outcome = (
                "matched" if actual_matched
                else "exception" if actual_exception
                else "unmatched"
            )

            if expected_outcome == "matched":
                if actual_matched and not actual_exception:
                    correct_decisions += 1
                    is_correct = True
                elif actual_exception:
                    false_positives += 1
            elif expected_outcome == "exception":
                if actual_exception and not actual_matched:
                    correct_decisions += 1
                    is_correct = True
                elif actual_matched:
                    false_negatives += 1

            match_method = None
            matched_batch_id = None
            exception_type = None
            severity = None
            amount_paise = None

            if actual_matched:
                mr = matched_by_id[record_id]
                match_method = mr.method.value if hasattr(mr.method, "value") else str(mr.method)
                matched_batch_id = mr.batch_id
                amount_paise = int(mr.amount_paise)

            if actual_exception:
                er = exceptions_by_id[record_id]
                exception_type = er.exception_type
                # severity lives on ClassifiedException objects in report.exceptions
                for ce in report.exceptions:
                    if ce.bank_credit_id == record_id:
                        severity = ce.severity
                        break
                if amount_paise is None:
                    ev = er.evidence if isinstance(er.evidence, dict) else {}
                    bp = ev.get("amount_paise") or ev.get("bank_amount_paise")
                    if isinstance(bp, int) and not isinstance(bp, bool):
                        amount_paise = bp

            records.append({
                "record_key": record_id,
                "trap_name": row.get("trap_name", ""),
                "amount_paise": amount_paise,
                "status": actual_outcome,
                "match_method": match_method,
                "matched_batch_id": matched_batch_id,
                "exception_type": exception_type,
                "severity": severity,
                "expected_outcome": expected_outcome,
                "correct": is_correct,
            })

        elif record_type == "settlement_batch":
            batch_id = record_id
            expected_type = row["expected_exception_type"]
            actual_outcome = "exception" if batch_id in classifier_by_batch else "none"

            if expected_outcome == "exception" and expected_type == "missing_settlement":
                if batch_id in classifier_by_batch:
                    exc = classifier_by_batch[batch_id]
                    if exc.exception_type == "missing_settlement":
                        correct_decisions += 1
                        is_correct = True

            records.append({
                "record_key": record_id,
                "trap_name": row.get("trap_name", ""),
                "amount_paise": None,
                "status": actual_outcome,
                "match_method": None,
                "matched_batch_id": None,
                "exception_type": (
                    classifier_by_batch[batch_id].exception_type
                    if batch_id in classifier_by_batch else None
                ),
                "severity": (
                    classifier_by_batch[batch_id].severity
                    if batch_id in classifier_by_batch else None
                ),
                "expected_outcome": expected_outcome,
                "correct": is_correct,
            })

        elif record_type == "order":
            order_id = record_id
            expected_type = row["expected_exception_type"]
            actual_outcome = "exception" if order_id in classifier_by_order else "none"

            if expected_outcome == "exception" and expected_type == "order_never_settled":
                if order_id in classifier_by_order:
                    exc = classifier_by_order[order_id]
                    if exc.exception_type == "order_never_settled":
                        correct_decisions += 1
                        is_correct = True

            records.append({
                "record_key": record_id,
                "trap_name": row.get("trap_name", ""),
                "amount_paise": None,
                "status": actual_outcome,
                "match_method": None,
                "matched_batch_id": None,
                "exception_type": (
                    classifier_by_order[order_id].exception_type
                    if order_id in classifier_by_order else None
                ),
                "severity": (
                    classifier_by_order[order_id].severity
                    if order_id in classifier_by_order else None
                ),
                "expected_outcome": expected_outcome,
                "correct": is_correct,
            })

    # Summary metrics
    total_bank_credits = sum(1 for r in gt_rows if r["record_type"] == "bank_credit")
    total_matched = len(all_matched)
    total_exceptions = len(report.exceptions)
    match_rate_pct = round(
        total_matched / total_bank_credits * 100, 1
    ) if total_bank_credits > 0 else 0.0

    deterministic_matches = sum(
        1 for m in all_matched
        if (m.method.value if hasattr(m.method, "value") else str(m.method)) in deterministic_methods
    )
    ai_assisted_matches = sum(
        1 for m in all_matched
        if (m.method.value if hasattr(m.method, "value") else str(m.method)) in ai_methods
    )

    accuracy_pct = round(
        correct_decisions / total_records * 100, 1
    ) if total_records > 0 else 0.0

    summary = {
        "total_bank_credits": total_bank_credits,
        "total_matched": total_matched,
        "total_exceptions": total_exceptions,
        "match_rate_pct": match_rate_pct,
        "deterministic_matches": deterministic_matches,
        "ai_assisted_matches": ai_assisted_matches,
        "accuracy_pct": accuracy_pct,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    return summary, records


# ---------------------------------------------------------------------------
# Step 4 — Assemble exceptions list for dashboard
# ---------------------------------------------------------------------------

def build_exceptions_list(report) -> list:
    exceptions = []
    for ce in report.exceptions:
        record_key = ce.bank_credit_id or ce.batch_id or ce.order_id or "unknown"
        exceptions.append({
            "record_key": record_key,
            "exception_type": ce.exception_type,
            "severity": ce.severity,
            "requires_human_review": ce.requires_human_review,
            "evidence": ce.evidence,
        })
    return exceptions


# ---------------------------------------------------------------------------
# Step 5 — Build audit trail from freshly written JSONL files
# ---------------------------------------------------------------------------

def build_audit_trail() -> list:
    audit_events = _read_all_audit_events(
        AUDIT_DIR / "audit_pass1.jsonl",
        AUDIT_DIR / "audit_pass2.jsonl",
        AUDIT_DIR / "audit_pass3.jsonl",
    )

    trail = []
    for ev in audit_events:
        trail.append({
            "seq": ev.get("seq", -1),
            "record_id": ev.get("bank_record_id", ""),
            "pass": _determine_pass(ev),
            "status": ev.get("status", ""),
            "reason": ev.get("reason", ""),
            "amount_paise": int(ev.get("actual_amount_paise", 0)),
            "metadata": ev.get("metadata") or {},
            "timestamp": ev.get("timestamp", ""),
        })
    return trail


# ---------------------------------------------------------------------------
# Step 6 — Razorpay touchpoint section
# ---------------------------------------------------------------------------

def build_razorpay_section() -> dict:
    """Fetch real Razorpay test-mode payments; gracefully handle missing keys."""
    key_id = os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
    keys_configured = bool(key_id and key_secret)

    print("Fetching Razorpay test-mode payments...")
    payments_raw = fetch_test_payments(count=10)  # never crashes

    payments_fetched = len(payments_raw)
    captured_count = sum(1 for p in payments_raw if p.get("status") == "captured")
    excluded_count = payments_fetched - captured_count

    payments_list = []
    for p in payments_raw:
        payments_list.append({
            "payment_id": p.get("id", ""),
            "amount_paise": int(p.get("amount", 0)),
            "status": p.get("status", ""),
            "included_in_settlement": p.get("status") == "captured",
        })

    return {
        "keys_configured": keys_configured,
        "payments_fetched": payments_fetched,
        "captured_count": captured_count,
        "excluded_count": excluded_count,
        "payments": payments_list,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DASHBOARD_DIR.mkdir(exist_ok=True)

    # 1. Clean stale output files for a fresh run
    clean_output_dir()

    # 2. Run full pipeline once
    all_matched, all_exceptions, report = run_pipeline()

    # 3. Compare against ground truth
    summary, records = compare_ground_truth(all_matched, all_exceptions, report)

    # 4. Exceptions list
    exceptions_list = build_exceptions_list(report)

    # 5. Audit trail
    audit_trail = build_audit_trail()

    # 6. Razorpay touchpoint
    razorpay_section = build_razorpay_section()

    # 7. Assemble final JSON object
    dashboard_data = {
        "generated_at": _utc_now_iso(),
        "summary": summary,
        "records": records,
        "exceptions": exceptions_list,
        "audit_trail": audit_trail,
        "razorpay_touchpoint": razorpay_section,
    }

    # 8. Write to dashboard/dashboard_data.json (pretty-printed)
    OUTPUT_PATH.write_text(
        json.dumps(dashboard_data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    n_records = len(records)
    n_exceptions = len(exceptions_list)
    n_audit = len(audit_trail)
    n_payments = len(razorpay_section["payments"])

    print(
        f"Dashboard data built: {n_records} records, {n_exceptions} exceptions, "
        f"{n_audit} audit events, {n_payments} Razorpay payments. "
        f"Written to dashboard/dashboard_data.json"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the Exception Classifier across the full SettleSense pipeline."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import json
from pathlib import Path

from matcher import (
    ExceptionRecord,
    MatchResult,
    load_settlement_batches_for_pass2,
    run_pass1,
    run_pass2,
    run_pass3,
)
from exception_classifier import classify_exceptions, ExceptionReport


def load_pass1_exceptions(audit_path: Path) -> list[ExceptionRecord]:
    """Reconstruct Pass 1 exceptions from the audit log (duplicate_credit)."""
    excs = []
    if not audit_path.exists():
        return excs
    with audit_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["status"] != "exception":
                continue
            etype = e.get("metadata", {}).get("exception_type", "unknown")
            excs.append(ExceptionRecord(
                bank_credit_id=e["bank_record_id"],
                exception_type=etype,
                evidence=e.get("metadata", {}),
            ))
    return excs


def main():
    data_dir = Path("data")
    audit_dir = Path("out")
    audit_dir.mkdir(exist_ok=True)

    # Full pipeline
    matched_p1, unmatched_p1 = run_pass1(
        data_dir / "bank_statement.csv",
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass1.jsonl",
    )
    newly_matched_p2, still_unmatched_p2, exceptions_p2 = run_pass2(
        unmatched_p1,
        matched_p1,
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass2.jsonl",
    )
    all_batches = load_settlement_batches_for_pass2(data_dir / "razorpay_settlements.csv")
    matched_batch_ids = {m.batch_id for m in matched_p1} | {m.batch_id for m in newly_matched_p2}
    batch_index = [b for b in all_batches if b.batch_id not in matched_batch_ids]
    newly_matched_p3, exceptions_p3 = run_pass3(
        still_unmatched_p2,
        batch_index,
        matched_p1 + newly_matched_p2,
    )

    exceptions_p1 = load_pass1_exceptions(audit_dir / "audit_pass1.jsonl")
    all_matched = matched_p1 + newly_matched_p2 + newly_matched_p3

    report = classify_exceptions(
        pass1_exceptions=exceptions_p1,
        pass2_exceptions=exceptions_p2,
        pass3_exceptions=exceptions_p3,
        all_matched=all_matched,
        settlement_path=str(data_dir / "razorpay_settlements.csv"),
        order_path=str(data_dir / "order_ledger.csv"),
        audit_path=str(audit_dir / "audit_exceptions.jsonl"),
    )

    all_pass = True
    excs = report.exceptions

    # Check 2: non-empty evidence
    print("--- Check 2: every exception has non-empty evidence ---")
    empty = [e.bank_credit_id or e.batch_id or e.order_id for e in excs if not e.evidence]
    if not empty:
        print("  PASS: all evidence dicts non-empty")
    else:
        print(f"  FAIL: empty evidence on {empty}")
        all_pass = False

    # Check 3: valid severity
    print("--- Check 3: valid severity ---")
    bad_sev = [e.exception_type for e in excs if e.severity not in ("high", "medium", "low")]
    if not bad_sev:
        print("  PASS: all severities valid")
    else:
        print(f"  FAIL: bad severity on {bad_sev}")
        all_pass = False

    # Check 4: requires_human_review is bool
    print("--- Check 4: requires_human_review is bool ---")
    bad_review = [e for e in excs if not isinstance(e.requires_human_review, bool)]
    if not bad_review:
        print("  PASS: all human-review flags are bool")
    else:
        print(f"  FAIL: {bad_review}")
        all_pass = False

    # Check 5: duplicate_credit has first_seen_in
    print("--- Check 5: duplicate_credit evidence has first_seen_in ---")
    dup = [e for e in excs if e.exception_type == "duplicate_credit"]
    dup_ok = all("first_seen_in" in e.evidence for e in dup)
    if dup and dup_ok:
        print(f"  PASS: first_seen_in={dup[0].evidence.get('first_seen_in')}")
    else:
        print(f"  FAIL: duplicate_credit count={len(dup)}")
        all_pass = False

    # Check 6: variance_breach has likely_cause
    print("--- Check 6: variance_breach evidence has likely_cause ---")
    var = [e for e in excs if e.exception_type == "variance_breach"]
    var_ok = all("likely_cause" in e.evidence for e in var)
    if var and var_ok:
        causes = {e.evidence["likely_cause"] for e in var}
        print(f"  PASS: causes={causes}")
    else:
        print(f"  FAIL: variance_breach count={len(var)}")
        all_pass = False

    # Check 7: settlement-side classification
    print("--- Check 7: order_never_settled count == 3 ---")
    ons_count = report.by_type.get("order_never_settled", 0)
    if ons_count == 3:
        print("  PASS: order_never_settled = 3")
    else:
        print(f"  FAIL: order_never_settled = {ons_count}")
        all_pass = False

    print("  (unmatched batches classified as missing_settlement):",
          report.by_type.get("missing_settlement", 0))

    # Check 8: no 'unknown' exception_type
    print("--- Check 8: no 'unknown' exception_type ---")
    unknown = [e for e in excs if e.exception_type == "unknown"]
    if not unknown:
        print("  PASS: all types are real")
    else:
        print(f"  FAIL: {unknown}")
        all_pass = False

    # -------- report table ----------
    print("\n=== EXCEPTION REPORT ===")
    print(f"{'type':<34} | {'severity':<7} | {'human':<5} | {'id':<12} | evidence")
    print("-" * 120)
    for e in excs:
        ident = e.bank_credit_id or e.batch_id or e.order_id or "-"
        key = " - "
        ev = e.evidence
        if e.exception_type == "duplicate_credit":
            key = f"dupe_of={ev.get('duplicate_of')}, first={ev.get('first_seen_in')}"
        elif e.exception_type == "variance_breach":
            key = f"delta={ev.get('delta_rupees')} cause={ev.get('likely_cause')}"
        elif e.exception_type == "ambiguous_amount":
            key = f"candidates={ev.get('candidate_count')}"
        elif e.exception_type == "utr_found_no_matching_batch":
            key = f"utr={ev.get('utr')}"
        elif e.exception_type == "ai_service_unavailable":
            key = "groq unavailable"
        elif e.exception_type == "missing_settlement":
            key = f"batch_net={ev.get('batch_net_paise')}, days={ev.get('days_since_settlement')}"
        elif e.exception_type == "zero_value_settlement":
            key = "zero"
        elif e.exception_type == "order_never_settled":
            key = f"amount={ev.get('amount_paise')}, days={ev.get('days_outstanding')}"
        print(f"{e.exception_type:<34} | {e.severity:<8} | {e.requires_human_review!s:<5} | {ident:<12} | {key}")

    print("\n" + "=" * 50)
    if all_pass:
        print("EXCEPTION CLASSIFIER VERIFIED ✓")
    else:
        print("EXCEPTION CLASSIFIER FAILED")
        for e in excs:
            if not e.evidence:
                print(f"  empty evidence -> {e.exception_type}")
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
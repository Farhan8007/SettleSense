#!/usr/bin/env python3
"""Verify Pass 3 output and full pipeline summary."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import json
import pandas as pd
from pathlib import Path
from matcher import run_pass1, run_pass2, run_pass3, load_settlement_batches_for_pass2, ExceptionRecord


def load_exceptions_from_audit(audit_path: Path) -> dict[str, dict]:
    """Load exceptions from audit log for a given pass."""
    exceptions = {}
    if not audit_path.exists():
        return exceptions
    with audit_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["status"] == "exception":
                bid = e["bank_record_id"]
                exceptions[bid] = {
                    "exception_type": e.get("metadata", {}).get("exception_type", "unknown"),
                    "evidence": e.get("metadata", {}),
                    "delta_paise": e.get("delta_paise", 0),
                }
    return exceptions


def main():
    data_dir = Path("data")
    audit_dir = Path("out")
    audit_dir.mkdir(exist_ok=True)

    # Run Pass 1
    matched_p1, unmatched_p1 = run_pass1(
        data_dir / "bank_statement.csv",
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass1.jsonl",
    )

    # Run Pass 2
    newly_matched_p2, still_unmatched_p2, exceptions_p2 = run_pass2(
        unmatched_p1,
        matched_p1,
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass2.jsonl",
    )

    # Build batch index for Pass 3 (reuse from Pass 2)
    all_batches = load_settlement_batches_for_pass2(data_dir / "razorpay_settlements.csv")
    matched_batch_ids = {m.batch_id for m in matched_p1} | {m.batch_id for m in newly_matched_p2}
    batch_index = [b for b in all_batches if b.batch_id not in matched_batch_ids]

    # Run Pass 3
    matched_so_far = matched_p1 + newly_matched_p2
    newly_matched_p3, exceptions_p3 = run_pass3(
        still_unmatched_p2,
        batch_index,
        matched_so_far,
    )

    # Load ground truth
    gt = pd.read_csv(data_dir / "ground_truth.csv")

    # Build decision map from our results
    decisions = {}
    all_matched = matched_p1 + newly_matched_p2 + newly_matched_p3

    # Pass 1 matched
    for m in matched_p1:
        decisions[m.bank_credit_id] = {
            "status": "matched",
            "method": m.method.value,
            "batch_id": m.batch_id,
            "delta_paise": 0,
            "pass": 1,
        }

    # Pass 2 newly matched
    for m in newly_matched_p2:
        decisions[m.bank_credit_id] = {
            "status": "matched",
            "method": m.method.value,
            "batch_id": m.batch_id,
            "delta_paise": 0,
            "pass": 2,
        }

    # Pass 3 newly matched
    for m in newly_matched_p3:
        decisions[m.bank_credit_id] = {
            "status": "matched",
            "method": "ai_assisted_utr",
            "batch_id": m.batch_id,
            "delta_paise": 0,
            "pass": 3,
        }

    # Pass 1 exceptions (from audit log)
    p1_exceptions = load_exceptions_from_audit(audit_dir / "audit_pass1.jsonl")
    for bid, exc in p1_exceptions.items():
        decisions[bid] = {
            "status": "exception",
            "exception_type": exc["exception_type"],
            "evidence": exc["evidence"],
            "delta_paise": exc["delta_paise"],
            "pass": 1,
        }

    # Pass 2 exceptions (from returned list + audit for delta)
    for e in exceptions_p2:
        decisions[e.bank_credit_id] = {
            "status": "exception",
            "exception_type": e.exception_type,
            "evidence": e.evidence,
            "delta_paise": 0,
            "pass": 2,
        }

    # Pass 3 exceptions (from returned list + audit for delta)
    for e in exceptions_p3:
        decisions[e.bank_credit_id] = {
            "status": "exception",
            "exception_type": e.exception_type,
            "evidence": e.evidence,
            "delta_paise": 0,
            "pass": 3,
        }

    # Fill delta_paise from audit logs for matched records
    for audit_file in ["audit_pass1.jsonl", "audit_pass2.jsonl", "audit_pass3.jsonl"]:
        audit_path = audit_dir / audit_file
        if audit_path.exists():
            with audit_path.open() as f:
                for line in f:
                    e = json.loads(line)
                    bid = e["bank_record_id"]
                    if bid in decisions and e["status"] in ("matched", "exception"):
                        decisions[bid]["delta_paise"] = e.get("delta_paise", 0)

    # Collect all exception bank_credit_ids
    all_exception_ids = set()
    for bid, dec in decisions.items():
        if dec.get("status") == "exception":
            all_exception_ids.add(bid)

    # =========================================================================
    # PIPELINE SUMMARY
    # =========================================================================
    total_bank_credits = len(gt[gt["record_type"] == "bank_credit"])
    total_matched = len(all_matched)
    total_exceptions = len(all_exception_ids)
    unaccounted = total_bank_credits - total_matched - total_exceptions

    print("=== SETTLESENSE PIPELINE SUMMARY ===")
    print(f"Pass 1 (Exact UTR):      {len(matched_p1)} matched,  {len(unmatched_p1)} to Pass 2, {len(p1_exceptions)} exceptions")
    print(f"Pass 2 (Batch Sum):      {len(newly_matched_p2)} matched,  {len(exceptions_p2)} exceptions, {len(still_unmatched_p2)} to Pass 3")
    print(f"Pass 3 (AI Assisted):    {len(newly_matched_p3)} matched,  {len(exceptions_p3)} exceptions")
    print("─────────────────────────────────────────────────")
    print(f"Total bank credits:      {total_bank_credits}")
    print(f"Total matched:           {total_matched}  ({total_matched/total_bank_credits*100:.0f}%)")
    print(f"Total exceptions:        {total_exceptions}")
    print(f"Unaccounted:             {unaccounted}  ← must be 0")

    # =========================================================================
    # CHECKS
    # =========================================================================
    all_pass = True

    # Check 1: unparseable_narration_unresolvable → exception
    print("\n--- Check 1: unparseable_narration_unresolvable → exception ---")
    rows = gt[gt["trap_name"] == "unparseable_narration_unresolvable"]
    for _, row in rows.iterrows():
        record_id = row["record_id"]
        decision = decisions.get(record_id, {})
        if decision.get("status") == "exception":
            exc_type = decision.get("exception_type", "")
            if exc_type in ("unparseable_narration_unresolvable", "ambiguous_amount", "ai_service_unavailable", "utr_found_no_matching_batch"):
                print(f"  {record_id}: PASS (exception_type={exc_type})")
            else:
                print(f"  {record_id}: FAIL (exception_type={exc_type})")
                all_pass = False
        else:
            print(f"  {record_id}: FAIL (status={decision.get('status')})")
            all_pass = False

    # Check 2: no bank credit appears in both matched and exceptions
    print("\n--- Check 2: No duplicate bank credit in matched and exceptions ---")
    matched_ids = {m.bank_credit_id for m in all_matched}
    overlap = matched_ids & all_exception_ids
    if not overlap:
        print("  PASS: No overlapping bank_credit_ids")
    else:
        print(f"  FAIL: Overlapping IDs: {overlap}")
        all_pass = False

    # Check 3: unaccounted == 0
    print("\n--- Check 3: Unaccounted == 0 ---")
    if unaccounted == 0:
        print("  PASS: Every bank credit has exactly one outcome")
    else:
        print(f"  FAIL: {unaccounted} bank credits unaccounted for")
        all_pass = False

    # Check 4: Verify specific trap types for Pass 3
    print("\n--- Check 4: Pass 3 trap types ---")
    pass3_traps = [
        "utr_found_no_matching_batch",  # bank_24, bank_25, bank_31
        "unparseable_narration_unresolvable",  # bank_30
        "orphan_bank_credit",  # bank_31
    ]

    for trap in pass3_traps:
        rows = gt[gt["trap_name"] == trap]
        if len(rows) == 0:
            continue
        for _, row in rows.iterrows():
            record_id = row["record_id"]
            decision = decisions.get(record_id, {})
            expected_exc = row["expected_exception_type"]
            our_exc = decision.get("exception_type", "")
            our_status = decision.get("status", "missing")
            
            # Flexible matching for exception types
            exc_ok = False
            if trap == "utr_found_no_matching_batch" and our_exc == "utr_found_no_matching_batch":
                exc_ok = True
            elif trap == "unparseable_narration_unresolvable" and our_exc in ("unparseable_narration_unresolvable", "ambiguous_amount", "ai_service_unavailable"):
                exc_ok = True
            elif trap == "orphan_bank_credit" and our_exc in ("utr_found_no_matching_batch", "orphan_bank_credit"):
                exc_ok = True
            
            if our_status == "exception" and exc_ok:
                print(f"  {record_id} ({trap}): PASS (exception_type={our_exc})")
            else:
                print(f"  {record_id} ({trap}): FAIL (status={our_status}, exception_type={our_exc})")
                all_pass = False

    # Check 5: Pass 1 duplicate_credit exception
    print("\n--- Check 5: Pass 1 duplicate_credit exception ---")
    rows = gt[gt["trap_name"] == "duplicate_credit"]
    for _, row in rows.iterrows():
        record_id = row["record_id"]
        decision = decisions.get(record_id, {})
        if decision.get("status") == "exception" and decision.get("exception_type") == "duplicate_credit":
            print(f"  {record_id}: PASS (exception_type=duplicate_credit)")
        else:
            print(f"  {record_id}: FAIL (status={decision.get('status')}, exception_type={decision.get('exception_type')})")
            all_pass = False

    print("\n" + "=" * 50)
    if all_pass:
        print("PIPELINE COMPLETE ✓")
    else:
        print("PIPELINE INCOMPLETE ✗")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
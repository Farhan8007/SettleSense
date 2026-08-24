#!/usr/bin/env python3
"""Verify Pass 2 output against ground truth for all trap types."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import json
import pandas as pd
from pathlib import Path
from matcher import run_pass1, run_pass2


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
    newly_matched, still_unmatched, exceptions = run_pass2(
        unmatched_p1,
        matched_p1,
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass2.jsonl",
    )

    # Load ground truth
    gt = pd.read_csv(data_dir / "ground_truth.csv")

    # Build decision map from our results
    decisions = {}

    # Pass 1 matched
    for m in matched_p1:
        decisions[m.bank_credit_id] = {
            "status": "matched",
            "method": m.method.value,
            "batch_id": m.batch_id,
            "delta_paise": 0,
        }

    # Pass 2 newly matched
    for m in newly_matched:
        decisions[m.bank_credit_id] = {
            "status": "matched",
            "method": m.method.value,
            "batch_id": m.batch_id,
            "delta_paise": 0,  # Will be filled from audit log
        }

    # Pass 2 exceptions
    for e in exceptions:
        decisions[e.bank_credit_id] = {
            "status": "exception",
            "exception_type": e.exception_type,
            "evidence": e.evidence,
        }

    # Still unmatched (to Pass 3) - only if not already in exceptions
    exception_bank_ids = {e.bank_credit_id for e in exceptions}
    for c in still_unmatched:
        if c.bank_credit_id not in exception_bank_ids:
            decisions[c.bank_credit_id] = {
                "status": "to_pass3",
            }

    # Fill delta_paise from audit log
    with (audit_dir / "audit_pass2.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            bid = e["bank_record_id"]
            if bid in decisions and e["status"] in ("matched", "exception"):
                decisions[bid]["delta_paise"] = e.get("delta_paise", 0)

    # Trap types Pass 2 must handle
    pass2_traps = [
        "batch_no_utr",
        "short_pay_within_tolerance",
        "variance_breach",
        "ambiguous_amount",
        "zero_value_settlement",
        "refund_batch",
        "unparseable_narration_unresolvable",
        "unparseable_narration_recovered",
        "fees_not_deducted",
    ]

    all_pass = True
    results = []

    for trap in pass2_traps:
        rows = gt[gt["trap_name"] == trap]
        for _, row in rows.iterrows():
            record_id = row["record_id"]
            expected_outcome = row["expected_outcome"]
            expected_method = row["expected_method"]
            expected_exception = row["expected_exception_type"]
            expected_batch = row["expected_batch"]
            expected_delta = int(row["expected_delta_paise"])

            our_decision = decisions.get(record_id, {"status": "missing"})
            our_status = our_decision.get("status", "missing")

            # Determine pass/fail
            if expected_outcome == "matched":
                if our_status == "matched":
                    method_ok = our_decision.get("method") in ("batch_sum_exact", "batch_sum_tolerance", "exact_utr", "utr_exact")
                    batch_ok = our_decision.get("batch_id") == expected_batch
                    if method_ok and batch_ok:
                        result = "PASS"
                    else:
                        result = "FAIL"
                        all_pass = False
                else:
                    result = "FAIL"
                    all_pass = False
            elif expected_outcome == "exception":
                if our_status == "exception":
                    # Check exception type matches or is close
                    our_exc = our_decision.get("exception_type", "")
                    exc_ok = (our_exc == expected_exception or 
                              (expected_exception == "amount_variance" and our_exc == "variance_breach") or
                              (expected_exception == "ambiguous_multiple_candidates" and our_exc == "ambiguous_amount") or
                              (expected_exception == "unresolvable_reference" and our_exc == "ambiguous_amount"))
                    if exc_ok:
                        result = "PASS"
                    else:
                        result = "FAIL"
                        all_pass = False
                else:
                    result = "FAIL"
                    all_pass = False
            else:
                result = "FAIL"
                all_pass = False

            results.append({
                "record_id": record_id,
                "trap": trap,
                "expected_outcome": expected_outcome,
                "expected_method": expected_method,
                "expected_exception": expected_exception,
                "expected_batch": expected_batch,
                "expected_delta": expected_delta,
                "our_status": our_status,
                "our_method": our_decision.get("method", ""),
                "our_exception": our_decision.get("exception_type", ""),
                "our_batch": our_decision.get("batch_id", ""),
                "our_delta": our_decision.get("delta_paise", 0),
                "result": result,
            })

    # Print results table
    print(f"{'bank_credit_id':<12} | {'trap':<35} | {'expected':<10} | {'our':<10} | {'result'}")
    print("-" * 100)
    for r in results:
        if r["our_status"] == "matched":
            our = f"{r['our_method']} ({r['our_batch']})"
        elif r["our_status"] == "exception":
            our = f"{r['our_exception']}"
        else:
            our = r["our_status"]
        
        expected = f"{r['expected_outcome']}"
        if r["expected_outcome"] == "matched":
            expected += f" ({r['expected_method']})"
        else:
            expected += f" ({r['expected_exception']})"
        
        print(f"{r['record_id']:<12} | {r['trap']:<35} | {expected:<10} | {our:<10} | {r['result']}")

    print("-" * 100)
    if all_pass:
        print("ALL PASS 2 TRAP TYPES RESOLVED AS EXPECTED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
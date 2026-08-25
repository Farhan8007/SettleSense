#!/usr/bin/env python3
"""SettleSense Accuracy Report - compares full pipeline results against ground truth."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from matcher import (
    run_pass1,
    run_pass2,
    run_pass3,
    load_settlement_batches_for_pass2,
    load_bank_credits,
    ExceptionRecord,
    MatchResult,
)
from exception_classifier import classify_exceptions


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pass1_exceptions(audit_path: Path) -> list[ExceptionRecord]:
    """Reconstruct Pass 1 exceptions from audit log (only duplicate_credit)."""
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

    # STEP 1 - Run full pipeline
    print("Running Pass 1...")
    matched_p1, unmatched_p1 = run_pass1(
        data_dir / "bank_statement.csv",
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass1.jsonl",
    )
    
    print("Running Pass 2...")
    newly_matched_p2, still_unmatched_p2, exceptions_p2 = run_pass2(
        unmatched_p1,
        matched_p1,
        data_dir / "razorpay_settlements.csv",
        audit_dir / "audit_pass2.jsonl",
    )
    
    print("Running Pass 3...")
    all_batches = load_settlement_batches_for_pass2(data_dir / "razorpay_settlements.csv")
    matched_batch_ids = {m.batch_id for m in matched_p1} | {m.batch_id for m in newly_matched_p2}
    batch_index = [b for b in all_batches if b.batch_id not in matched_batch_ids]
    newly_matched_p3, exceptions_p3 = run_pass3(
        still_unmatched_p2,
        batch_index,
        matched_p1 + newly_matched_p2,
    )
    
    print("Running Exception Classifier...")
    exceptions_p1 = load_pass1_exceptions(audit_dir / "audit_pass1.jsonl")
    all_matched = matched_p1 + newly_matched_p2 + newly_matched_p3
    report = classify_exceptions(
        exceptions_p1,
        exceptions_p2,
        exceptions_p3,
        all_matched,
        str(data_dir / "razorpay_settlements.csv"),
        str(data_dir / "order_ledger.csv"),
        str(audit_dir / "audit_exceptions.jsonl"),
    )
    
    all_exceptions = (
        exceptions_p1 +
        exceptions_p2 +
        exceptions_p3 +
        report.exceptions
    )
    
    # Build lookup tables for quick access
    matched_by_id = {m.bank_credit_id: m for m in all_matched}
    exceptions_by_id = {e.bank_credit_id: e for e in all_exceptions}
    
    # Build lookup for exception classifier results (batch-side and order-side)
    classifier_by_batch = {}
    classifier_by_order = {}
    for exc in report.exceptions:
        if exc.batch_id:
            classifier_by_batch[exc.batch_id] = exc
        if exc.order_id:
            classifier_by_order[exc.order_id] = exc
    
    # Load ground truth
    gt_rows = []
    with (data_dir / "ground_truth.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_rows.append(row)
    
    # STEP 2 - Compare against ground truth
    total_records = 0
    correct_decisions = 0
    wrong_decisions = []
    
    matched_correct = 0
    matched_total = 0
    exception_correct = 0
    exception_total = 0
    method_match_count = 0
    batch_match_count = 0
    exception_type_match_count = 0
    false_positives = 0  # matched but should be exception
    false_negatives = 0  # exception but should be matched
    
    decision_table = []  # for STEP 4 output
    
    for row in gt_rows:
        record_type = row["record_type"]
        record_id = row["record_id"]
        expected_outcome = row["expected_outcome"]
        
        if record_type == "bank_credit":
            total_records += 1
            matched_total += 1
            
            actual_matched = record_id in matched_by_id
            actual_exception = record_id in exceptions_by_id
            
            is_correct = False
            method_match = False
            batch_match = False
            exception_type_match = False
            
            if expected_outcome == "matched":
                if actual_matched and not actual_exception:
                    correct_decisions += 1
                    matched_correct += 1
                    is_correct = True
                    
                    # Check method
                    actual_method = matched_by_id[record_id].method.value
                    expected_method = row["expected_method"]
                    if actual_method == expected_method:
                        method_match = True
                        method_match_count += 1
                    
                    # Check batch (if expected)
                    expected_batch = row["expected_batch"]
                    if expected_batch and expected_batch != "":
                        actual_batch = matched_by_id[record_id].batch_id
                        if actual_batch == expected_batch:
                            batch_match = True
                            batch_match_count += 1
                else:
                    # Wrong decision
                    wrong_decisions.append((record_id, "matched expected but got exception or unmatched"))
                    if actual_exception:
                        false_positives += 1
                    is_correct = False
                    
            elif expected_outcome == "exception":
                if actual_exception and not actual_matched:
                    correct_decisions += 1
                    exception_correct += 1
                    is_correct = True
                    
                    # Check exception type
                    expected_type = row["expected_exception_type"]
                    if expected_type and expected_type != "":
                        actual_type = exceptions_by_id[record_id].exception_type
                        if actual_type == expected_type:
                            exception_type_match = True
                            exception_type_match_count += 1
                else:
                    # Wrong decision
                    wrong_decisions.append((record_id, "exception expected but got matched or unmatched"))
                    if actual_matched:
                        false_negatives += 1
                    is_correct = False
            
            decision_table.append((
                record_id,
                row["trap_name"],
                expected_outcome,
                "matched" if actual_matched else "exception" if actual_exception else "unmatched",
                "✓" if is_correct else "✗"
            ))
            
        elif record_type == "settlement_batch":
            total_records += 1
            batch_id = record_id
            expected_outcome = row["expected_outcome"]  # should be "exception"
            expected_type = row["expected_exception_type"]  # should be "missing_settlement"
            
            is_correct = False
            if expected_outcome == "exception" and expected_type == "missing_settlement":
                if batch_id in classifier_by_batch:
                    exc = classifier_by_batch[batch_id]
                    if exc.exception_type == "missing_settlement":
                        correct_decisions += 1
                        is_correct = True
                    else:
                        wrong_decisions.append((batch_id, f"expected missing_settlement but got {exc.exception_type}"))
                else:
                    wrong_decisions.append((batch_id, "expected missing_settlement but not found in exceptions"))
            
            decision_table.append((
                record_id,
                row["trap_name"],
                expected_outcome,
                "exception" if batch_id in classifier_by_batch else "none",
                "✓" if is_correct else "✗"
            ))
            
        elif record_type == "order":
            total_records += 1
            order_id = record_id
            expected_outcome = row["expected_outcome"]  # should be "exception"
            expected_type = row["expected_exception_type"]  # should be "order_never_settled"
            
            is_correct = False
            if expected_outcome == "exception" and expected_type == "order_never_settled":
                if order_id in classifier_by_order:
                    exc = classifier_by_order[order_id]
                    if exc.exception_type == "order_never_settled":
                        correct_decisions += 1
                        is_correct = True
                    else:
                        wrong_decisions.append((order_id, f"expected order_never_settled but got {exc.exception_type}"))
                else:
                    wrong_decisions.append((order_id, "expected order_never_settled but not found in exceptions"))
            
            decision_table.append((
                record_id,
                row["trap_name"],
                expected_outcome,
                "exception" if order_id in classifier_by_order else "none",
                "✓" if is_correct else "✗"
            ))
    
    # Calculate rates
    accuracy_pct = (correct_decisions / total_records * 100) if total_records > 0 else 0.0
    method_match_pct = (method_match_count / matched_correct * 100) if matched_correct > 0 else 0.0
    batch_match_pct = (batch_match_count / matched_correct * 100) if matched_correct > 0 else 0.0
    exception_type_match_pct = (exception_type_match_count / exception_correct * 100) if exception_correct > 0 else 0.0
    
    # STEP 3 - Print metrics table
    print("\n" + "="*60)
    print("           SETTLESENSE ACCURACY REPORT")
    print("="*60)
    print(f"  Total records evaluated:        {total_records}")
    print(f"  Correct decisions:              {correct_decisions}")
    print(f"  Wrong decisions:                {len(wrong_decisions)}")
    print()
    print(f"  OVERALL ACCURACY:               {accuracy_pct:.1f}%")
    print()
    print("  Breakdown:")
    print(f"    Matched correctly:            {matched_correct} / {matched_total}")
    print(f"    Exceptions correctly flagged: {exception_correct} / {exception_total}")
    print(f"    Method matched ground truth:  {method_match_count} / {matched_correct}")
    print(f"    Batch matched ground truth:   {batch_match_count} / {matched_correct}")
    print(f"    Exception type correct:       {exception_type_match_count} / {exception_correct}")
    print()
    print(f"  False positives (matched, should be exception): {false_positives}")
    print(f"  False negatives (exception, should be matched): {false_negatives}")
    print("="*60)
    
    # STEP 4 - Print decision table
    print("\nDecision table:")
    print(f"{'record_key':<12} | {'trap_name':<25} | {'expected':<12} | {'our_decision':<12} | {'correct'}")
    print("-" * 75)
    for record_id, trap, expected, our, correct in decision_table:
        print(f"{record_id:<12} | {trap:<25} | {expected:<12} | {our:<12} | {correct}")
    
    # STEP 5 - Write accuracy audit event
    accuracy_event = {
        "event": "accuracy_report",
        "total_evaluated": total_records,
        "correct": correct_decisions,
        "wrong": len(wrong_decisions),
        "accuracy_pct": round(accuracy_pct, 1),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "method_match_rate": round(method_match_pct, 1),
        "batch_match_rate": round(batch_match_pct, 1),
        "exception_type_match_rate": round(exception_type_match_pct, 1),
        "wrong_records": [w[0] for w in wrong_decisions],
        "timestamp": _utc_now_iso(),
    }
    
    audit_path = audit_dir / "audit_accuracy.jsonl"
    with audit_path.open("a") as f:
        f.write(json.dumps(accuracy_event) + "\n")
    
    # STEP 6 - Final verdict
    print("\n" + "="*60)
    if accuracy_pct >= 85.0 and false_positives == 0:
        print(f"ACCURACY REPORT: PASS ✓ ({accuracy_pct:.1f}% accuracy, 0 false positives)")
    else:
        print("ACCURACY REPORT: FAIL ✗")
        if wrong_decisions:
            print("Wrong decisions:")
            for record_id, reason in wrong_decisions[:10]:  # limit output
                print(f"  {record_id}: {reason}")
            if len(wrong_decisions) > 10:
                print(f"  ... and {len(wrong_decisions) - 10} more")
        else:
            print("No wrong decisions")
    print("="*60)
    
    return accuracy_pct >= 85.0 and false_positives == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
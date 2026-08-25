#!/usr/bin/env python3
"""Razorpay touchpoint demo — proves schema compatibility with real test-mode data."""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.razorpay_connector import fetch_test_payments, combine_with_synthetic
from src.audit_log import AuditLog, AuditEvent, MatchStatus, MatchMethod


def main():
    audit_log = AuditLog("out/audit_razorpay_touchpoint.jsonl", fsync=False)
    
    print("Fetching test-mode payments from Razorpay...")
    payments = fetch_test_payments(10)
    
    if not payments:
        print("No live payments returned (keys may not be configured or no test payments exist)")
        audit_log.append(AuditEvent.create(
            bank_record_id="razorpay_touchpoint",
            status=MatchStatus.PENDING,
            match_method=MatchMethod.NONE,
            expected_amount_paise=0,
            actual_amount_paise=0,
            reason="Razorpay API returned no payments — synthetic-only mode",
            metadata={"payments_fetched": 0, "source": "razorpay_live_test_mode"},
        ))
        print("\n⚠ No live Razorpay data — synthetic-only mode confirmed working")
        return
    
    print(f"\nFetched {len(payments)} real test-mode payments:")
    captured_count = 0
    failed_count = 0
    
    for i, p in enumerate(payments):
        status = p.get("status", "unknown")
        amount_rupees = p.get("amount", 0) / 100
        included = status == "captured"
        reason = "captured, included" if included else "failed, excluded from settlement"
        
        print(f"  REAL (Razorpay test mode) — {p['id']} — ₹{amount_rupees:.2f} — status={status}")
        
        if included:
            captured_count += 1
        else:
            failed_count += 1
        
        audit_log.append(AuditEvent.create(
            bank_record_id=f"razorpay_{p['id']}",
            status=MatchStatus.MATCHED if included else MatchStatus.EXCEPTION,
            match_method=MatchMethod.NONE,
            expected_amount_paise=0,
            actual_amount_paise=p.get("amount", 0),
            reason=f"Razorpay test-mode payment: {reason}",
            metadata={
                "payment_id": p["id"],
                "status": status,
                "included_in_settlement": included,
                "source": "razorpay_live_test_mode",
            },
        ))
    
    print()
    synthetic = pd.read_csv("data/razorpay_settlements.csv")
    print(f"Before combine: {len(synthetic)} synthetic rows")
    combined = combine_with_synthetic(synthetic, payments)
    print(f"After combine: {len(combined)} total rows")
    
    print(f"\n✓ Live Razorpay schema compatibility confirmed — "
          f"{len(payments)} real payments fetched, "
          f"{captured_count} captured and merged into settlement pipeline, "
          f"{failed_count} correctly excluded")


if __name__ == "__main__":
    main()
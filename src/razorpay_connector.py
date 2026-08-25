def fetch_test_payments(count: int = 10) -> list[dict]:
    """Fetch real test-mode payments from Razorpay. Never crashes —
    returns empty list on any failure."""
    import razorpay
    import os
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        print("⚠ Razorpay keys not set — skipping fetch")
        return []
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        response = client.payment.all({"count": count})
        return response.get("items", [])
    except Exception as e:
        print(f"⚠ Razorpay API unavailable ({e})")
        return []


def normalize_razorpay_payment(payment: dict) -> dict:
    import datetime
    
    amount_paise = payment["amount"]
    if not isinstance(amount_paise, int):
        raise ValueError(f"Expected int paise, got {type(amount_paise)}")
    
    captured_amount = payment.get("amount_captured")
    if captured_amount is None:
        captured_amount = amount_paise if payment.get("captured") else 0
    
    fee_paise = payment.get("fee") or 0
    tax_paise = payment.get("tax") or 0
    net_paise = captured_amount - fee_paise - tax_paise
    
    acquirer = payment.get("acquirer_data") or {}
    utr = acquirer.get("utr") or acquirer.get("transaction_id")
    
    created_ts = payment.get("created_at")
    settled_at = (datetime.datetime.fromtimestamp(created_ts, tz=datetime.timezone.utc)
                  .strftime("%Y-%m-%d") if created_ts else "")
    
    return {
        "settlement_id": f"live_{payment['id']}",
        "payment_id": payment["id"],
        "order_id": payment.get("order_id", ""),
        "amount": amount_paise,
        "fee": fee_paise,
        "gst_on_fee": tax_paise,
        "net_amount": net_paise,
        "settled_at": settled_at,
        "status": payment.get("status", ""),
        "utr": utr,
        "source": "razorpay_live_test_mode"
    }


def combine_with_synthetic(synthetic_df, live_payments: list[dict]):
    """Mix up to 5 CAPTURED live payments into the synthetic
    settlements DataFrame. Failed payments are excluded — they never
    reach settlement in reality. Returns combined DataFrame."""
    import pandas as pd
    
    captured_only = [p for p in live_payments if p.get("status") == "captured"]
    skipped = len(live_payments) - len(captured_only)
    if skipped:
        print(f"Excluded {skipped} non-captured payment(s) from settlement mix-in")
    
    normalized = [normalize_razorpay_payment(p) for p in captured_only[:5]]
    if not normalized:
        print("No captured live payments to mix in — synthetic-only")
        return synthetic_df
    
    live_df = pd.DataFrame(normalized)
    combined = pd.concat([synthetic_df, live_df], ignore_index=True)
    print(f"Mixed in {len(normalized)} real captured Razorpay test-mode "
          f"payments alongside {len(synthetic_df)} synthetic records")
    return combined
"""Generate the SettleSense synthetic dataset — with ground truth DERIVED.

Why this file exists
--------------------
The previous dataset was hand-written, and hand-written ground truth drifts from
hand-written data. An audit of the original CSVs found that only 2 of 33 bank
credits actually equalled their settlement batch's net amount (and both of those
were the zero-rupee rows). 22 credits equalled the batch GROSS — meaning fees and
GST were never deducted — and 9 matched neither gross nor net. The ground-truth
file also referenced a `bank_unparseable` row that did not exist in the bank
statement, and its `razorpay_NN` ids were off by one.

The fix is structural, not a patch: build the data FORWARD
(orders -> payments -> settlement batches -> bank credits) using the same
settlement_math module the matcher uses, so a bank credit is *by construction*
the sum of its batch's per-payment net amounts. Then inject each hard case as a
deliberate, named deviation from that correct baseline, and emit ground truth as
a byproduct of the construction. Ground truth can then never disagree with the
data, because it is not written independently of it.

Run:  python src/generate_data.py          (writes into ./data)
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from settlement_math import (
    DEFAULT_TOLERANCE_PAISE,
    PaymentMath,
    compute_payment,
    roll_up,
    to_paise,
    to_rupees,
)

SEED = 20260823  # fixed so the dataset is byte-reproducible


# ---------------------------------------------------------------------------
# Trap catalogue. Each name is a deliberate, documented deviation from a
# correctly-reconciling baseline. The matcher is scored on whether it reaches
# the expected outcome for each of these.
# ---------------------------------------------------------------------------

TRAPS = {
    "clean_single": "One payment, UTR present in narration. Should match exactly.",
    "clean_batch": "Several payments bundled, UTR present. Batch net must be summed.",
    "batch_no_utr": "Bundled payments, narration has no UTR. Must fall back to amount+date window.",
    "duplicate_credit": "Identical credit appears twice. Second must NOT consume the same batch.",
    "missing_settlement": "Batch exists in Razorpay but never reached the bank.",
    "orphan_bank_credit": "Bank credit with no corresponding Razorpay batch.",
    "short_pay_within_tolerance": "Credit short by 50 paise — inside declared tolerance, match with variance recorded.",
    "variance_breach": "Credit short by a material amount — must NOT auto-match.",
    "unparseable_narration_recovered": "Narration carries no usable reference, but the net is unique in the window so the amount fallback recovers it.",
    "unparseable_narration_unresolvable": "Narration carries no usable reference AND the amount fits several candidates. Genuinely undecidable — human review.",
    "zero_value_settlement": "Zero-rupee batch and zero-rupee credit. Must not divide by zero or match by amount alone.",
    "refund_batch": "Negative batch (refunds). Fee and GST reverse too.",
    "ambiguous_amount": "Two UTR-less batches with identical net in the same window. Must report ambiguity, not guess.",
    "fees_not_deducted": "Credit equals batch GROSS — upstream feed bug, not a match.",
}

CATEGORIES = ("electronics", "apparel", "home", "books", "beauty")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class Payment:
    payment_id: str
    order_id: str
    customer_id: str
    gross: int                 # paise
    math: PaymentMath
    status: str                # captured | refunded
    created_at: datetime
    category: str
    settled: bool = True       # False => order exists but never settled


@dataclass
class Batch:
    settlement_id: str
    payments: list[Payment]
    settled_at: datetime
    utr: str                   # what Razorpay reports; "" if none
    reaches_bank: bool = True

    @property
    def math(self):
        return roll_up(
            [p.math for p in self.payments], [p.payment_id for p in self.payments]
        )


@dataclass
class BankRow:
    date: datetime
    narration: str
    amount: int                # paise
    ref_number: str
    # ground-truth provenance, not written to the bank CSV
    trap: str = "clean_single"
    batch_id: str = ""
    expected_outcome: str = "matched"
    expected_method: str = "utr_exact"
    expected_exception: str = ""
    expected_delta: int = 0
    notes: str = ""


@dataclass
class Dataset:
    payments: list[Payment] = field(default_factory=list)
    batches: list[Batch] = field(default_factory=list)
    bank_rows: list[BankRow] = field(default_factory=list)
    unpaid_orders: list[Payment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class Builder:
    def __init__(self, seed: int = SEED) -> None:
        self.rng = random.Random(seed)
        self.ds = Dataset()
        self._pay_n = 0
        self._ord_n = 0
        self._setl_n = 0
        self._day = datetime(2024, 1, 1, 9, 0)

    # -- id helpers --------------------------------------------------------
    def _next_payment_ids(self) -> tuple[str, str, str]:
        self._pay_n += 1
        self._ord_n += 1
        return (
            f"pay_{self._pay_n:03d}",
            f"ord_{self._ord_n:03d}",
            f"cust_{self._ord_n:03d}",
        )

    def _next_settlement_id(self) -> str:
        self._setl_n += 1
        return f"setl_{self._setl_n:03d}"

    def _advance(self, hours: int = 0) -> datetime:
        self._day += timedelta(hours=hours or self.rng.choice([14, 18, 22, 26]))
        return self._day

    def _utr(self, kind: str = "NEFT") -> str:
        body = "".join(self.rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8))
        return f"{kind}{body}"

    # -- payment / batch construction --------------------------------------
    def _make_payment(self, gross_rupees: str, when: datetime, status: str = "captured") -> Payment:
        pid, oid, cid = self._next_payment_ids()
        gross = to_paise(gross_rupees)
        return Payment(
            payment_id=pid,
            order_id=oid,
            customer_id=cid,
            gross=gross,
            math=compute_payment(gross),
            status=status,
            created_at=when - timedelta(hours=self.rng.randint(4, 30)),
            category=self.rng.choice(CATEGORIES),
        )

    def _make_batch(
        self,
        gross_list: list[str],
        utr: str = "",
        status: str = "captured",
        reaches_bank: bool = True,
    ) -> Batch:
        when = self._advance()
        payments = [self._make_payment(g, when, status) for g in gross_list]
        batch = Batch(
            settlement_id=self._next_settlement_id(),
            payments=payments,
            settled_at=when,
            utr=utr,
            reaches_bank=reaches_bank,
        )
        self.ds.payments.extend(payments)
        self.ds.batches.append(batch)
        return batch

    # -- bank row construction ---------------------------------------------
    def _credit(
        self,
        batch: Batch,
        narration_style: str,
        trap: str,
        *,
        amount_override: int | None = None,
        expected_outcome: str = "matched",
        expected_method: str = "utr_exact",
        expected_exception: str = "",
        notes: str = "",
        date_offset_hours: int = 2,
    ) -> BankRow:
        """Emit a bank credit for a batch. Default amount is the batch NET."""
        expected_net = batch.math.net
        amount = expected_net if amount_override is None else amount_override
        ref, narration = self._narration(narration_style, batch)
        row = BankRow(
            date=batch.settled_at + timedelta(hours=date_offset_hours),
            narration=narration,
            amount=amount,
            ref_number=ref,
            trap=trap,
            batch_id=batch.settlement_id,
            expected_outcome=expected_outcome,
            expected_method=expected_method,
            expected_exception=expected_exception,
            expected_delta=amount - expected_net,
            notes=notes or TRAPS.get(trap, ""),
        )
        self.ds.bank_rows.append(row)
        return row

    def _narration(self, style: str, batch: Batch) -> tuple[str, str]:
        """Return (ref_number, narration). Only 'neft'/'rtgs' expose the UTR."""
        if style == "neft":
            return f"NEFT{batch.utr}", f"NEFT-RAZORPAY-{batch.utr}"
        if style == "rtgs":
            return f"RTGS{batch.utr}", f"RTGS-RAZORPAY-{batch.utr}"
        if style == "imps":
            ref = f"IMPS{self.rng.randint(100000, 999999)}"
            return ref, f"IMPS/{ref[4:]}/RAZORPAY-SETTLEMENT"
        if style == "upi":
            ref = f"UPI{self.rng.randint(100000, 999999)}"
            return ref, f"UPI/CR/{self.rng.randint(10**9, 10**10 - 1)}/RAZORPAY"
        if style == "garbled":
            return "", f"MISC CR {self.rng.randint(1000000, 9999999)} CLG"
        raise ValueError(style)


# ---------------------------------------------------------------------------
# The scenario: a month of settlements with every trap deliberately placed
# ---------------------------------------------------------------------------

def build() -> Dataset:
    b = Builder()

    # --- a run of clean, correctly-reconciling activity ------------------
    # These carry the pipeline's credibility: most real records must just work.
    for gross, style in [
        ("24500.00", "upi"),
        ("31200.00", "neft"),
        ("125000.00", "rtgs"),
        ("18900.00", "imps"),
        ("42100.00", "upi"),
        ("38900.00", "neft"),
        ("9800.00", "imps"),
        ("27300.00", "upi"),
        ("95000.00", "rtgs"),
        ("22100.00", "neft"),
        ("48900.00", "imps"),
        ("16500.00", "upi"),
        ("53400.00", "neft"),
        ("14200.00", "imps"),
        ("62300.00", "neft"),
    ]:
        utr = b._utr("RTGS" if style == "rtgs" else "UTR") if style in ("neft", "rtgs") else ""
        batch = b._make_batch([gross], utr=utr)
        b._credit(
            batch, style, "clean_single",
            expected_method="utr_exact" if utr else "amount_window",
        )

    # --- clean multi-payment batches with a UTR to key off ---------------
    for grosses in (
        ["25000.00", "12000.00", "8000.00", "3500.00", "1200.00"],
        ["40000.00", "20000.00", "8000.00"],
        ["50000.00", "35000.00"],
        ["30000.00", "15000.00", "7000.00"],
    ):
        batch = b._make_batch(grosses, utr=b._utr("UTR"))
        b._credit(batch, "neft", "clean_batch", expected_method="batch_by_utr")

    # --- batches with NO UTR in the narration: amount + window fallback --
    for grosses in (["30000.00", "25000.00"], ["21000.00", "11500.00", "4300.00"]):
        batch = b._make_batch(grosses)
        b._credit(batch, "imps", "batch_no_utr", expected_method="amount_window")

    # --- TRAP: duplicate credit -----------------------------------------
    dup_batch = b._make_batch(["50000.00", "25000.00", "15000.00"], utr=b._utr("UTR"))
    b._credit(dup_batch, "neft", "clean_batch", expected_method="batch_by_utr")
    dup = b._credit(
        dup_batch, "neft", "duplicate_credit",
        expected_outcome="exception",
        expected_method="none",
        expected_exception="duplicate_bank_credit",
        notes="Same UTR and amount as the preceding row. The batch is already "
              "consumed, so this must be flagged, not matched again.",
    )
    dup.ref_number = f"NEFT{dup_batch.utr}"  # identical ref — the giveaway

    # --- TRAP: settlement that never reached the bank --------------------
    for gross in ("18500.00", "22000.00", "33000.00"):
        b._make_batch([gross], utr=b._utr("UTR"), reaches_bank=False)

    # --- TRAP: short-paid but inside tolerance --------------------------
    tol_batch = b._make_batch(["17650.00"], utr=b._utr("UTR"))
    b._credit(
        tol_batch, "neft", "short_pay_within_tolerance",
        amount_override=tol_batch.math.net - 50,
        expected_method="utr_exact",
        notes="50 paise short. Inside the ₹1.00 declared tolerance, so it may "
              "match — but the delta must be recorded, not swallowed.",
    )

    # --- TRAP: variance beyond tolerance --------------------------------
    var_batch = b._make_batch(["64000.00"], utr=b._utr("UTR"))
    b._credit(
        var_batch, "neft", "variance_breach",
        amount_override=var_batch.math.net - to_paise("2500.00"),
        expected_outcome="exception",
        expected_method="none",
        expected_exception="amount_variance",
        notes="₹2,500 short against a resolvable UTR. Reference matches but "
              "money does not — the most dangerous case to auto-match.",
    )

    # --- TRAP: fees never deducted upstream (credit == gross) -----------
    gross_batch = b._make_batch(["47600.00", "12400.00"], utr=b._utr("UTR"))
    b._credit(
        gross_batch, "neft", "fees_not_deducted",
        amount_override=gross_batch.math.gross,
        expected_outcome="exception",
        expected_method="none",
        expected_exception="amount_variance",
        notes="Credit equals batch GROSS. reconcile() should attach the "
              "'fees not deducted' diagnostic rather than report a blind delta.",
    )

    # --- TRAP: zero-value settlement ------------------------------------
    zero_batch = b._make_batch(["0.00"], utr=b._utr("UTR"))
    b._credit(
        zero_batch, "neft", "zero_value_settlement",
        expected_method="utr_exact",
        notes="Zero on both sides. Matching on amount alone would pair this "
              "with any other zero row, so the reference must carry the match.",
    )

    # --- TRAP: refund batch (negative) ----------------------------------
    refund_batch = b._make_batch(["-2000.00", "-300.00"], utr=b._utr("UTR"), status="refunded")
    b._credit(
        refund_batch, "neft", "refund_batch",
        expected_method="batch_by_utr",
        notes="Negative batch: fee and GST reverse with the principal.",
    )

    # --- TRAP: two UTR-less batches with identical net, same window -----
    amb_a = b._make_batch(["19000.00", "6000.00"])
    amb_b_ = b._make_batch(["19000.00", "6000.00"])
    amb_b_.settled_at = amb_a.settled_at + timedelta(hours=3)
    for bt in (amb_a, amb_b_):
        b._credit(
            bt, "imps", "ambiguous_amount",
            expected_outcome="exception",
            expected_method="none",
            expected_exception="ambiguous_multiple_candidates",
            notes="Two candidate batches reconcile equally well and neither "
                  "narration carries a UTR. Guessing would be a coin flip.",
        )

    # --- TRAP: unparseable narration, TWO distinct cases -----------------
    # Case 1: reference unreadable AND amount ambiguous -> genuinely
    # undecidable, must go to human review. This is the graceful-failure demo.
    amb_c = b._make_batch(["19000.00", "6000.00"])
    amb_c.settled_at = amb_a.settled_at + timedelta(hours=6)
    b._credit(
        amb_c, "garbled", "unparseable_narration_unresolvable",
        expected_outcome="exception",
        expected_method="none",
        expected_exception="unresolvable_reference",
        notes="No readable reference AND the net matches several candidate "
              "batches in the window. Nothing can decide this correctly, so "
              "refusing is the right answer.",
    )

    # Case 2: reference unreadable but the net is unique in the window, so the
    # amount fallback legitimately recovers it. Scoring this as an exception
    # would punish a matcher for being right, so ground truth says 'matched'.
    unp_batch = b._make_batch(["28750.00"])
    b._credit(
        unp_batch, "garbled", "unparseable_narration_recovered",
        expected_method="amount_window",
        notes="Regex extraction fails, amount fallback succeeds. The audit log "
              "must show BOTH the extractor miss and the fallback that rescued "
              "it — the recovery is the interesting part, not the match.",
    )

    # --- TRAP: orphan bank credit (no batch at all) ---------------------
    orphan = BankRow(
        date=datetime(2024, 1, 29, 11, 30),
        narration="NEFT-OTHERBANK-UTRZZ9911QQ",
        amount=to_paise("7350.00"),
        ref_number="NEFTZZ9911QQ",
        trap="orphan_bank_credit",
        batch_id="",
        expected_outcome="exception",
        expected_method="none",
        expected_exception="orphan_bank_credit",
        expected_delta=0,
        notes="Credit from a source that is not Razorpay at all.",
    )
    b.ds.bank_rows.append(orphan)

    # --- orders that never produced a settlement ------------------------
    # Without these the order ledger is a perfect 1:1 mirror of the settlement
    # feed, and the third source proves nothing.
    for gross, status in (("5400.00", "failed"), ("12750.00", "failed"), ("8900.00", "pending")):
        p = b._make_payment(gross, datetime(2024, 1, 20, 12, 0), status=status)
        p.settled = False
        b.ds.unpaid_orders.append(p)

    b.ds.bank_rows.sort(key=lambda r: r.date)
    return b.ds


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_all(ds: Dataset, out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}

    # bank_statement.csv -- ids are assigned by row order to match
    # normalizer's bank_{idx}; ref_number is carried into ground truth as a
    # cross-check so index drift is detectable by a test rather than silently
    # scoring the wrong row.
    running = to_paise("125000.00")
    with (out / "bank_statement.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "narration", "amount", "balance", "ref_number"])
        for r in ds.bank_rows:
            running += r.amount
            w.writerow([
                r.date.strftime("%Y-%m-%d"), r.narration,
                f"{to_rupees(r.amount):.2f}", f"{to_rupees(running):.2f}", r.ref_number,
            ])
    counts["bank_statement"] = len(ds.bank_rows)

    with (out / "razorpay_settlements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["settlement_id", "payment_id", "order_id", "amount", "fee",
                    "gst_on_fee", "net_amount", "settled_at", "status", "utr"])
        n = 0
        for batch in ds.batches:
            for p in batch.payments:
                w.writerow([
                    batch.settlement_id, p.payment_id, p.order_id,
                    f"{to_rupees(p.gross):.2f}", f"{to_rupees(p.math.fee):.2f}",
                    f"{to_rupees(p.math.gst_on_fee):.2f}", f"{to_rupees(p.math.net):.2f}",
                    batch.settled_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "refunded" if p.status == "refunded" else "processed",
                    batch.utr,
                ])
                n += 1
    counts["razorpay_settlements"] = n

    with (out / "order_ledger.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "payment_id", "customer_id", "amount", "status",
                    "created_at", "product_category"])
        n = 0
        for p in ds.payments + ds.unpaid_orders:
            w.writerow([
                p.order_id, p.payment_id, p.customer_id, f"{to_rupees(p.gross):.2f}",
                "refunded" if p.status == "refunded"
                else ("completed" if p.settled else p.status),
                p.created_at.strftime("%Y-%m-%d %H:%M:%S"), p.category,
            ])
            n += 1
    counts["order_ledger"] = n

    # ground_truth.csv -- fixed 11 columns on every row, derived from the build
    with (out / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "record_type", "record_key", "expected_outcome",
                    "expected_method", "expected_batch", "expected_payment_ids",
                    "expected_exception_type", "expected_delta_paise", "trap_name",
                    "notes"])
        n = 0
        for i, r in enumerate(ds.bank_rows):
            pay_ids = ""
            if r.batch_id:
                batch = next(b for b in ds.batches if b.settlement_id == r.batch_id)
                pay_ids = "|".join(p.payment_id for p in batch.payments)
            w.writerow([
                f"bank_{i}", "bank_credit", r.ref_number, r.expected_outcome,
                r.expected_method, r.batch_id, pay_ids, r.expected_exception,
                r.expected_delta, r.trap, r.notes,
            ])
            n += 1
        for batch in ds.batches:
            if batch.reaches_bank:
                continue
            w.writerow([
                batch.settlement_id, "settlement_batch", batch.settlement_id,
                "exception", "none", batch.settlement_id,
                "|".join(p.payment_id for p in batch.payments),
                "missing_settlement", 0, "missing_settlement",
                TRAPS["missing_settlement"],
            ])
            n += 1
        for p in ds.unpaid_orders:
            w.writerow([
                p.order_id, "order", p.order_id, "exception", "none", "", "",
                "order_never_settled", 0, "order_never_settled",
                f"Order in status '{p.status}' with no settlement. Expected: no "
                f"bank credit, and not counted as a reconciliation failure.",
            ])
            n += 1
    counts["ground_truth"] = n

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    args = ap.parse_args()

    ds = build()
    counts = write_all(ds, Path(args.out))

    reaching = [b for b in ds.batches if b.reaches_bank]
    print(f"Wrote to {Path(args.out).resolve()}")
    for k, v in counts.items():
        print(f"  {k+'.csv':<28} {v:>4} rows")
    print(f"\n  settlement batches           {len(ds.batches):>4} "
          f"({len(reaching)} reaching the bank)")
    print(f"  payments                     {len(ds.payments):>4}")
    print(f"  orders never settled         {len(ds.unpaid_orders):>4}")
    print(f"  declared tolerance           {DEFAULT_TOLERANCE_PAISE} paise")

    traps: dict[str, int] = {}
    for r in ds.bank_rows:
        traps[r.trap] = traps.get(r.trap, 0) + 1
    traps["missing_settlement"] = len(ds.batches) - len(reaching)
    traps["order_never_settled"] = len(ds.unpaid_orders)
    print("\n  trap distribution:")
    for name, count in sorted(traps.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<34} {count:>3}")


if __name__ == "__main__":
    main()

"""Settlement money identity — the single source of truth for SettleSense.

Everything that touches money goes through this module: the data generator uses
it to build correct bank credits, and the matcher uses it to decide whether a
credit reconciles. Because both sides share one implementation, the generator
and the matcher cannot silently disagree.

THE IDENTITY
------------
For one payment:

    fee   = round(gross * 1.8%)          # Razorpay platform fee
    gst   = round(fee   * 18%)           # GST is levied on the fee, not the sale
    net   = gross - fee - gst            # what Razorpay actually remits

For a settlement batch (many payments bundled into one bank credit):

    bank_credit = sum(net for each payment in the batch)

Two details that are easy to get wrong and that a payments person will check:

1. Rounding is PER PAYMENT, not on the batch total. Razorpay computes and
   rounds the fee and GST for each payment individually, then remits the sum.
   Rounding the batch gross instead can shift the answer by a rupee or two,
   which is exactly the sort of unexplained delta reconciliation exists to
   catch. See `test_settlement_math.py::test_per_payment_rounding_differs`.

2. All money is handled as integer PAISE, never float rupees. 0.1 + 0.2 != 0.3
   in binary floating point, and a reconciliation engine that accumulates float
   error will manufacture deltas that do not exist.

Refunds carry through with negative gross, and therefore negative fee, GST and
net — a refund returns the fee too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Rates. Change them here and both the generator and the matcher follow.
# ---------------------------------------------------------------------------

FEE_RATE = Decimal("0.018")   # 1.8% platform fee on gross
GST_RATE = Decimal("0.18")    # 18% GST, charged on the fee

#: Default acceptable variance when comparing a bank credit to expected net.
#: One rupee. Deliberately small: tolerance hides money, so it must be a
#: declared policy rather than an accident, and every use of it is logged.
DEFAULT_TOLERANCE_PAISE = 100


# ---------------------------------------------------------------------------
# Rupee <-> paise conversion
# ---------------------------------------------------------------------------

def to_paise(rupees: str | int | float | Decimal) -> int:
    """Convert a rupee amount to integer paise, half-up at the paise boundary.

    Accepts str/Decimal (preferred, exact) as well as int/float for
    convenience when reading CSVs. Floats are routed through str() so that
    47832.00 does not arrive as 47831.999999.
    """
    if isinstance(rupees, float):
        rupees = str(rupees)
    d = Decimal(rupees) * 100
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def to_rupees(paise: int) -> Decimal:
    """Convert integer paise to an exact Decimal rupee amount (2 dp)."""
    return (Decimal(paise) / 100).quantize(Decimal("0.01"))


def fmt(paise: int) -> str:
    """Human-readable rupee string for logs and reports, e.g. '-₹1,957.52'."""
    sign = "-" if paise < 0 else ""
    return f"{sign}₹{abs(to_rupees(paise)):,.2f}"


def _pct(paise: int, rate: Decimal) -> int:
    """Apply a rate to a paise amount, rounding half-up, sign-symmetric.

    Rounding is applied to the magnitude so that a refund of -X incurs exactly
    the negative of the fee charged on +X. Rounding the signed value directly
    would make refunds fail to net out to zero on half-paise boundaries.
    """
    magnitude = (Decimal(abs(paise)) * rate).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(magnitude) if paise >= 0 else -int(magnitude)


# ---------------------------------------------------------------------------
# Per-payment math
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaymentMath:
    """The fee breakdown for a single payment. All fields integer paise."""

    gross: int
    fee: int
    gst_on_fee: int
    net: int

    def __post_init__(self) -> None:
        assert self.net == self.gross - self.fee - self.gst_on_fee, (
            f"PaymentMath violates identity: {self.gross} - {self.fee} "
            f"- {self.gst_on_fee} != {self.net}"
        )


def compute_payment(gross_paise: int) -> PaymentMath:
    """Compute fee, GST and net for one payment from its gross amount."""
    fee = _pct(gross_paise, FEE_RATE)
    gst = _pct(fee, GST_RATE)
    return PaymentMath(
        gross=gross_paise, fee=fee, gst_on_fee=gst, net=gross_paise - fee - gst
    )


# ---------------------------------------------------------------------------
# Batch rollup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchMath:
    """Rolled-up totals for a settlement batch. All fields integer paise."""

    payment_count: int
    gross: int
    fee: int
    gst_on_fee: int
    net: int
    payment_ids: tuple[str, ...] = ()

    def explain(self) -> str:
        """One-line human explanation — this is what goes in the audit log."""
        return (
            f"{self.payment_count} payment(s) grossing {fmt(self.gross)} "
            f"less fees {fmt(self.fee)} less GST on fees {fmt(self.gst_on_fee)} "
            f"= {fmt(self.net)} expected net"
        )


def roll_up(
    payments: Iterable[PaymentMath], payment_ids: Sequence[str] = ()
) -> BatchMath:
    """Sum per-payment math into batch totals.

    Note this sums *already-rounded* per-payment values, which is what Razorpay
    remits. Do not be tempted to compute the fee on the batch gross instead.
    """
    payments = list(payments)
    return BatchMath(
        payment_count=len(payments),
        gross=sum(p.gross for p in payments),
        fee=sum(p.fee for p in payments),
        gst_on_fee=sum(p.gst_on_fee for p in payments),
        net=sum(p.net for p in payments),
        payment_ids=tuple(payment_ids),
    )


def roll_up_from_gross(
    gross_amounts: Iterable[int], payment_ids: Sequence[str] = ()
) -> BatchMath:
    """Convenience: roll up a batch straight from gross paise amounts."""
    return roll_up([compute_payment(g) for g in gross_amounts], payment_ids)


# ---------------------------------------------------------------------------
# Reconciliation verdict
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Outcome of comparing a bank credit to a batch's expected net."""

    EXACT = "exact"                      # delta is zero to the paise
    WITHIN_TOLERANCE = "within_tolerance"  # non-zero but inside declared policy
    VARIANCE = "variance"                # outside tolerance — do not auto-match


@dataclass(frozen=True)
class Reconciliation:
    """Result of reconciling one bank credit against one candidate batch.

    `delta` is signed: positive means the bank received MORE than expected,
    negative means short-paid. Sign matters — a short-payment is a very
    different investigation from an over-credit, and collapsing to abs()
    throws away the only clue about which.
    """

    bank_credit: int
    expected: BatchMath
    delta: int
    verdict: Verdict
    tolerance_paise: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_match(self) -> bool:
        """True if this credit may be auto-matched under current policy."""
        return self.verdict in (Verdict.EXACT, Verdict.WITHIN_TOLERANCE)

    def explain(self) -> str:
        """Full human-readable reconciliation line for the audit log."""
        head = f"Bank credit {fmt(self.bank_credit)} vs {self.expected.explain()}"
        if self.verdict is Verdict.EXACT:
            return f"{head} — exact to the paise."
        direction = "over-credited" if self.delta > 0 else "short-paid"
        tail = (
            f"within declared tolerance of {fmt(self.tolerance_paise)}"
            if self.verdict is Verdict.WITHIN_TOLERANCE
            else f"EXCEEDS tolerance of {fmt(self.tolerance_paise)} — not auto-matched"
        )
        return f"{head} — {direction} by {fmt(abs(self.delta))}, {tail}."

    def to_audit_dict(self) -> dict:
        """Structured form for the JSONL audit log."""
        return {
            "bank_credit_paise": self.bank_credit,
            "expected_gross_paise": self.expected.gross,
            "expected_fee_paise": self.expected.fee,
            "expected_gst_paise": self.expected.gst_on_fee,
            "expected_net_paise": self.expected.net,
            "delta_paise": self.delta,
            "payment_count": self.expected.payment_count,
            "payment_ids": list(self.expected.payment_ids),
            "verdict": self.verdict.value,
            "tolerance_paise": self.tolerance_paise,
            "explanation": self.explain(),
            "notes": self.notes,
        }


def reconcile(
    bank_credit_paise: int,
    expected: BatchMath,
    tolerance_paise: int = DEFAULT_TOLERANCE_PAISE,
) -> Reconciliation:
    """Compare a bank credit against a candidate batch's expected net.

    This never decides *which* batch to compare against — that is the matcher's
    job. This function only answers "given this candidate, do the numbers work,
    and by how much are they off." Keeping the arithmetic separate from the
    candidate search is what makes both testable.
    """
    delta = bank_credit_paise - expected.net
    if delta == 0:
        verdict = Verdict.EXACT
    elif abs(delta) <= tolerance_paise:
        verdict = Verdict.WITHIN_TOLERANCE
    else:
        verdict = Verdict.VARIANCE

    notes: list[str] = []
    # Diagnostic that pays for itself in a panel Q&A: if the credit equals the
    # batch GROSS, fees were never deducted — that is a data or upstream bug,
    # not a genuine reconciliation break, and saying so is worth a lot.
    if bank_credit_paise == expected.gross and expected.fee != 0:
        notes.append(
            "credit equals batch GROSS — fees and GST appear not to have been "
            "deducted upstream; investigate the settlement feed, not the match"
        )

    return Reconciliation(
        bank_credit=bank_credit_paise,
        expected=expected,
        delta=delta,
        verdict=verdict,
        tolerance_paise=tolerance_paise,
        notes=notes,
    )

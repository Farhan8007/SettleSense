"""Verify the GENERATED dataset. Fails loudly if the data does not reconcile.

This is the guard the original hand-written dataset lacked. It re-derives every
expected value from the CSVs independently of the generator's internal state, so
it catches drift rather than restating the generator's assumptions.

Run from the repo root:
    python src/generate_data.py            # regenerate ./data
    python tests/check_dataset.py          # verify ./data
    python tests/check_dataset.py /tmp/gen # verify some other directory
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from settlement_math import (  # noqa: E402
    DEFAULT_TOLERANCE_PAISE, compute_payment, fmt, reconcile, roll_up, to_paise,
)

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
failures: list[str] = []
checks = 0


def check(cond: bool, msg: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def rows(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def raw_field_counts(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return {len(r) for r in csv.reader(f)}


bank = rows("bank_statement.csv")
setl = rows("razorpay_settlements.csv")
orders = rows("order_ledger.csv")
truth = rows("ground_truth.csv")

print("=" * 78)
print("A. CSV well-formedness (the original ground_truth.csv had ragged rows)")
print("=" * 78)
for name in ("bank_statement.csv", "razorpay_settlements.csv",
             "order_ledger.csv", "ground_truth.csv"):
    widths = raw_field_counts(name)
    check(len(widths) == 1, f"{name} has ragged rows: field counts {sorted(widths)}")
    print(f"  {name:<28} field counts {sorted(widths)}  "
          f"{'OK' if len(widths) == 1 else 'RAGGED'}")

print()
print("=" * 78)
print("B. Per-payment identity: net == gross - fee - gst, fee == 1.8%, gst == 18%")
print("=" * 78)
bad = 0
for r in setl:
    g, fee, gst, net = (to_paise(r[k]) for k in ("amount", "fee", "gst_on_fee", "net_amount"))
    exp = compute_payment(g)
    if (fee, gst, net) != (exp.fee, exp.gst_on_fee, exp.net):
        bad += 1
        if bad <= 5:
            print(f"  {r['payment_id']}: got fee={fee} gst={gst} net={net} "
                  f"expected fee={exp.fee} gst={exp.gst_on_fee} net={exp.net}")
check(bad == 0, f"{bad} settlement rows violate the per-payment identity")
print(f"  {len(setl)} rows checked, {bad} violations")

print()
print("=" * 78)
print("C. THE BIG ONE: does each bank credit equal its batch NET?")
print("=" * 78)
batch_pay = defaultdict(list)
for r in setl:
    batch_pay[r["settlement_id"]].append(r)

gt_by_id = {t["record_id"]: t for t in truth}
matched_expected = variance_expected = 0
print(f"  {'bank':<9}{'credit':>13}{'batch':<11}{'expected_net':>14}{'delta':>11}"
      f"  {'gt_delta':>9}  outcome")
print("  " + "-" * 74)
for i, r in enumerate(bank):
    rid = f"bank_{i}"
    gt = gt_by_id.get(rid)
    check(gt is not None, f"{rid} has no ground-truth row")
    if gt is None:
        continue
    # index/ref cross-check: catches the off-by-one that broke the old GT
    check(gt["record_key"] == r["ref_number"],
          f"{rid} ref mismatch: bank has {r['ref_number']!r}, "
          f"ground truth has {gt['record_key']!r}")

    credit = to_paise(r["amount"])
    bid = gt["expected_batch"]
    if not bid:
        print(f"  {rid:<9}{fmt(credit):>13}{'(none)':<11}{'-':>14}{'-':>11}"
              f"  {gt['expected_delta_paise']:>9}  {gt['expected_exception_type']}")
        continue

    bm = roll_up([compute_payment(to_paise(p['amount'])) for p in batch_pay[bid]],
                 [p["payment_id"] for p in batch_pay[bid]])
    rec = reconcile(credit, bm)
    delta = rec.delta
    check(delta == int(gt["expected_delta_paise"]),
          f"{rid} delta {delta} != ground truth {gt['expected_delta_paise']}")

    if gt["expected_outcome"] == "matched":
        matched_expected += 1
        check(rec.is_match,
              f"{rid} ground truth says matched but reconcile() says "
              f"{rec.verdict.value} (delta {fmt(delta)})")
    else:
        variance_expected += 1

    print(f"  {rid:<9}{fmt(credit):>13}{bid:<11}{fmt(bm.net):>14}{fmt(delta):>11}"
          f"  {gt['expected_delta_paise']:>9}  {gt['expected_outcome']}"
          f"/{rec.verdict.value}")

print("  " + "-" * 74)
print(f"  expected to match: {matched_expected}   expected to be an exception: "
      f"{variance_expected}")

print()
print("=" * 78)
print("D. Ground-truth referential integrity")
print("=" * 78)
setl_ids = {r["settlement_id"] for r in setl}
pay_ids = {r["payment_id"] for r in setl}
order_ids = {r["order_id"] for r in orders}
bank_ids = {f"bank_{i}" for i in range(len(bank))}
for t in truth:
    rt = t["record_type"]
    if rt == "bank_credit":
        check(t["record_id"] in bank_ids,
              f"GT references {t['record_id']} but bank has only "
              f"bank_0..bank_{len(bank)-1}")
    elif rt == "settlement_batch":
        check(t["record_id"] in setl_ids, f"GT references unknown batch {t['record_id']}")
    elif rt == "order":
        check(t["record_id"] in order_ids, f"GT references unknown order {t['record_id']}")
    for p in filter(None, t["expected_payment_ids"].split("|")):
        check(p in pay_ids, f"GT row {t['record_id']} references unknown payment {p}")
print(f"  {len(truth)} ground-truth rows, all ids resolved against the CSVs")

print()
print("=" * 78)
print("E. Traps are actually present in the data")
print("=" * 78)
present = defaultdict(int)
for t in truth:
    present[t["trap_name"]] += 1
required = ["duplicate_credit", "missing_settlement", "orphan_bank_credit",
            "short_pay_within_tolerance", "variance_breach",
            "unparseable_narration_recovered", "unparseable_narration_unresolvable",
            "zero_value_settlement", "refund_batch", "ambiguous_amount",
            "fees_not_deducted", "order_never_settled", "batch_no_utr"]
for name in required:
    n = present.get(name, 0)
    check(n > 0, f"trap '{name}' is declared but has no rows in the data")
    print(f"  {name:<38}{n:>3} row(s)  {'OK' if n else 'MISSING'}")

# the 'unparseable' rows must genuinely defeat the normalizer's extractors
PATTERNS = [r'UTR([A-Z0-9]+)', r'IMPS/(\d+)/RAZORPAY', r'UPI/CR/(\d+)/RAZORPAY']
for t in truth:
    if not t["trap_name"].startswith("unparseable_narration"):
        continue
    idx = int(t["record_id"].split("_")[1])
    nar = bank[idx]["narration"]
    hit = [p for p in PATTERNS if re.search(p, nar, re.I)]
    check(not hit, f"'unparseable' narration {nar!r} is actually matched by {hit}")
    print(f"  {nar!r} defeats all {len(PATTERNS)} extractors: "
          f"{'OK' if not hit else 'FAILS'}")

# the unresolvable case must ALSO be ambiguous by amount, otherwise the amount
# fallback would legitimately resolve it and 'exception' would be the wrong label
nets = defaultdict(list)
for bid, prows in batch_pay.items():
    n = roll_up([compute_payment(to_paise(p["amount"])) for p in prows]).net
    nets[n].append(bid)
for t in truth:
    if t["trap_name"] != "unparseable_narration_unresolvable":
        continue
    credit = to_paise(bank[int(t["record_id"].split("_")[1])]["amount"])
    n_cand = len(nets.get(credit, []))
    check(n_cand > 1,
          f"{t['record_id']} is labelled unresolvable but only {n_cand} batch(es) "
          f"match its amount — the amount fallback would resolve it correctly")
    print(f"  {t['record_id']} amount {fmt(credit)} matches {n_cand} candidate "
          f"batches -> genuinely undecidable: {'OK' if n_cand > 1 else 'FAILS'}")

print()
print("=" * 78)
print("F. Third source earns its place (order ledger must NOT mirror settlements)")
print("=" * 78)
unsettled = [o for o in orders if o["payment_id"] not in pay_ids]
check(len(unsettled) > 0,
      "order ledger is a 1:1 mirror of settlements — the third source adds nothing")
print(f"  orders: {len(orders)}   settlement rows: {len(setl)}   "
      f"orders with no settlement: {len(unsettled)}")
for o in unsettled:
    print(f"    {o['order_id']} status={o['status']} amount={o['amount']}")

print()
print("=" * 78)
if failures:
    print(f"FAILED — {len(failures)} of {checks} checks")
    for m in failures:
        print(f"  x {m}")
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED — the dataset reconciles by construction.")
print(f"declared tolerance: {DEFAULT_TOLERANCE_PAISE} paise")

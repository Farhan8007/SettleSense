#!/usr/bin/env python3
"""Step 8 — Failure Handling Demo for buildathon pitch video.

Runs a fresh, complete reconciliation pipeline (Pass 1 → 2 → 3 →
classify_exceptions) and then narrates 3 specific failures with real
data, showing how each is handled gracefully without crashing,
guessing, or silently failing.

Designed to be run LIVE on camera. Safe to re-run multiple times —
always starts from a clean pipeline state, never mutates source data.

Usage:
    python3 demo/failure_demo.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
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
)
from exception_classifier import classify_exceptions
from audit_log import AuditLog, AuditEvent, MatchStatus, MatchMethod
from settlement_math import fmt

DATA_DIR = PROJECT_ROOT / "data"
AUDIT_DIR = PROJECT_ROOT / "out"
DEMO_AUDIT_PATH = AUDIT_DIR / "audit_failure_demo.jsonl"

# Width of the separator line
SEP = "─" * 56


# ---------------------------------------------------------------------------
# Pipeline helpers (mirrors dashboard/build_dashboard_data.py)
# ---------------------------------------------------------------------------

def _load_pass1_exceptions(audit_path: Path) -> list:
    """Reconstruct Pass 1 exceptions from audit log."""
    excs = []
    if not audit_path.exists():
        return excs
    with audit_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("status") != "exception":
                continue
            etype = e.get("metadata", {}).get(
                "exception_type", "unknown"
            )
            excs.append(ExceptionRecord(
                bank_credit_id=e["bank_record_id"],
                exception_type=etype,
                evidence=e.get("metadata", {}),
            ))
    return excs


def clean_output_dir() -> None:
    """Remove stale JSONL + lock files for a fresh run."""
    AUDIT_DIR.mkdir(exist_ok=True)
    for pattern in ("*.jsonl", "*.jsonl.lock"):
        for f in AUDIT_DIR.glob(pattern):
            f.unlink()


def run_pipeline():
    """Run Pass 1 → 2 → 3 → classify_exceptions."""
    print("  Running Pass 1 (exact UTR)…")
    matched_p1, unmatched_p1 = run_pass1(
        DATA_DIR / "bank_statement.csv",
        DATA_DIR / "razorpay_settlements.csv",
        AUDIT_DIR / "audit_pass1.jsonl",
    )

    print("  Running Pass 2 (batch sum)…")
    newly_matched_p2, still_unmatched_p2, exceptions_p2 = run_pass2(
        unmatched_p1,
        matched_p1,
        DATA_DIR / "razorpay_settlements.csv",
        AUDIT_DIR / "audit_pass2.jsonl",
    )

    print("  Running Pass 3 (AI-assisted narration)…")
    all_batches = load_settlement_batches_for_pass2(
        DATA_DIR / "razorpay_settlements.csv"
    )
    matched_batch_ids = (
        {m.batch_id for m in matched_p1}
        | {m.batch_id for m in newly_matched_p2}
    )
    batch_index = [
        b for b in all_batches
        if b.batch_id not in matched_batch_ids
    ]
    newly_matched_p3, exceptions_p3 = run_pass3(
        still_unmatched_p2,
        batch_index,
        matched_p1 + newly_matched_p2,
    )

    print("  Running Exception Classifier…")
    exceptions_p1 = _load_pass1_exceptions(
        AUDIT_DIR / "audit_pass1.jsonl"
    )
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

    return report


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _pp_evidence(evidence: dict) -> str:
    """Pretty-print evidence dict, indented for terminal."""
    raw = json.dumps(evidence, indent=2, ensure_ascii=False)
    lines = raw.split("\n")
    return "\n".join("     " + line for line in lines)


def _cause_label(cause: str) -> str:
    """Human-readable label for a likely_cause code."""
    labels = {
        "fees_not_deducted": "fees not deducted",
        "extra_deduction": "extra deduction applied",
        "partial_payment": "partial payment",
        "unknown": "unknown cause",
    }
    return labels.get(cause, cause)


# ---------------------------------------------------------------------------
# Narration blocks
# ---------------------------------------------------------------------------

def narrate_duplicate(idx: int, total: int, exc) -> list[str]:
    """Build narration lines for a duplicate_credit failure."""
    ev = exc.evidence
    utr = ev.get("utr", "???")
    first = ev.get("first_seen_in", "???")
    amount = ev.get("amount_paise", 0)

    lines = [
        SEP,
        f"FAILURE {idx} of {total}: "
        f"Duplicate Credit ({exc.bank_credit_id})",
        SEP,
        "",
        "What happened:",
        f"  Bank statement shows UTR {utr} appearing",
        f"  TWICE — once as {first} (already matched),",
        f"  once as {exc.bank_credit_id}.",
        "",
        "What SettleSense did:",
        "  ✓ Matched the first occurrence normally",
        f"  ✓ Flagged {exc.bank_credit_id} as a duplicate"
        " — did NOT",
        "    count it as revenue",
        "  ✓ Logged the decision with full evidence"
        " to the",
        "    audit trail",
        "",
        "Why this matters:",
        f"  Silently matching both would have",
        f"  double-counted {fmt(amount)} in revenue.",
        "  A human now reviews this instead.",
        "",
        "Full audit evidence:",
        _pp_evidence(ev),
        SEP,
    ]
    return lines


def narrate_variance(idx: int, total: int, exc) -> list[str]:
    """Build narration lines for a variance_breach failure."""
    ev = exc.evidence
    candidate = ev.get("candidate_batch_id", "???")
    bank_amt = ev.get("bank_amount_paise", 0)
    batch_net = ev.get("batch_net_paise", 0)
    delta = ev.get("delta_paise", 0)
    delta_rupees = ev.get("delta_rupees", fmt(delta))
    cause = ev.get("likely_cause", "unknown")

    lines = [
        SEP,
        f"FAILURE {idx} of {total}: "
        f"Variance Breach ({exc.bank_credit_id})",
        SEP,
        "",
        "What happened:",
        f"  Bank credit {fmt(bank_amt)} does not match",
        f"  nearest batch {candidate} "
        f"(net {fmt(batch_net)}).",
        f"  Delta: {delta_rupees}",
        "",
        "What SettleSense did:",
        "  ✓ Identified the closest candidate batch",
        f"  ✓ Computed exact delta: {delta_rupees}",
        f"  ✓ Diagnosed likely cause: "
        f"{_cause_label(cause)}",
        "  ✓ Routed to human review — did NOT force"
        " a match",
        "",
        "Why this matters:",
        "  Blindly matching with a "
        f"{delta_rupees} gap would",
        "  hide a real discrepancy. The exact delta",
        "  and cause give a reviewer a head start.",
        "",
        "Full audit evidence:",
        _pp_evidence(ev),
        SEP,
    ]
    return lines


def narrate_ambiguous(idx: int, total: int, exc) -> list[str]:
    """Build narration lines for an ambiguous_amount failure."""
    ev = exc.evidence
    bank_amt = ev.get("bank_amount_paise", 0)
    candidates = ev.get("candidate_batch_ids", [])
    count = ev.get("candidate_count", len(candidates))

    batch_list = ", ".join(candidates) if candidates else "???"

    lines = [
        SEP,
        f"FAILURE {idx} of {total}: "
        f"Ambiguous Amount ({exc.bank_credit_id})",
        SEP,
        "",
        "What happened:",
        f"  Bank credit {fmt(bank_amt)} matches {count}",
        "  settlement batches within the date window",
        "  — genuinely undecidable without more info.",
        f"  Competing batches: {batch_list}",
        "",
        "What SettleSense did:",
        f"  ✓ Found all {count} candidate matches",
        "  ✓ Refused to guess — flagged as ambiguous",
        "  ✓ Routed to human review with all"
        f" {count}",
        "    candidates named",
        "",
        "Why this matters:",
        "  Picking one at random would create a",
        "  confident-looking match that might be wrong.",
        "  A human sees all candidates and decides.",
        "",
        "Full audit evidence:",
        _pp_evidence(ev),
        SEP,
    ]
    return lines


# ---------------------------------------------------------------------------
# Demo audit logging
# ---------------------------------------------------------------------------

def log_demo_failures(failures: list) -> None:
    """Write each demonstrated failure to the demo audit log."""
    # Remove stale demo log for idempotent re-runs
    if DEMO_AUDIT_PATH.exists():
        DEMO_AUDIT_PATH.unlink()

    log = AuditLog(str(DEMO_AUDIT_PATH), fsync=True)

    for exc in failures:
        ev = exc.evidence
        amount = (
            ev.get("amount_paise")
            or ev.get("bank_amount_paise")
            or 0
        )
        event = AuditEvent.create(
            bank_record_id=exc.bank_credit_id,
            status=MatchStatus.EXCEPTION,
            match_method=MatchMethod.NONE,
            expected_amount_paise=0,
            actual_amount_paise=amount,
            reason=(
                f"Demo narration: {exc.exception_type} — "
                f"severity {exc.severity}"
            ),
            metadata={
                "exception_type": exc.exception_type,
                "severity": exc.severity,
                "demo_run": True,
                "narrated": True,
                "evidence": {
                    k: v for k, v in ev.items()
                    if not str(k).endswith("_paise")
                },
            },
        )
        log.append(event)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Intro --
    print()
    print("SettleSense — Failure Handling Demo")
    print(
        "Running the full reconciliation pipeline,"
        " then examining 3"
    )
    print(
        "records that could NOT be cleanly matched"
        " — and showing how"
    )
    print(
        "each is handled without crashing, guessing,"
        " or silently"
    )
    print("failing.\n")

    # -- Step 1: Clean + run pipeline --
    print("Step 1: Fresh pipeline run")
    clean_output_dir()
    report = run_pipeline()
    print(f"  Pipeline complete: "
          f"{report.total_exceptions} exceptions classified.\n")

    # -- Step 2: Find our 3 target failures --
    exc_by_id = {}
    for exc in report.exceptions:
        if exc.bank_credit_id:
            exc_by_id[exc.bank_credit_id] = exc

    # 1. bank_22 — duplicate_credit
    dup = exc_by_id.get("bank_22")

    # 2. bank_24 or bank_25 — variance_breach
    var = exc_by_id.get("bank_24")
    if var is None or var.exception_type != "variance_breach":
        var = exc_by_id.get("bank_25")

    # 3. bank_30 — ambiguous_amount
    amb = exc_by_id.get("bank_30")

    failures = [dup, var, amb]
    found = [f for f in failures if f is not None]

    if len(found) < 3:
        missing = []
        if dup is None:
            missing.append("duplicate_credit (bank_22)")
        if var is None:
            missing.append(
                "variance_breach (bank_24/bank_25)"
            )
        if amb is None:
            missing.append("ambiguous_amount (bank_30)")
        print(
            f"⚠ Only {len(found)} of 3 target failures "
            f"found. Missing: {', '.join(missing)}"
        )
        print(
            "  (Pipeline output may have changed."
            " Narrating what we have.)\n"
        )
        failures = found

    total = len(failures)

    # -- Step 3: Narrate each failure --
    print(f"Step 2: Narrating {total} failure(s)\n")

    for i, exc in enumerate(failures, 1):
        if exc.exception_type == "duplicate_credit":
            lines = narrate_duplicate(i, total, exc)
        elif exc.exception_type == "variance_breach":
            lines = narrate_variance(i, total, exc)
        elif exc.exception_type == "ambiguous_amount":
            lines = narrate_ambiguous(i, total, exc)
        else:
            # Fallback for unexpected types
            lines = [
                SEP,
                f"FAILURE {i} of {total}: "
                f"{exc.exception_type} "
                f"({exc.bank_credit_id})",
                SEP,
                _pp_evidence(exc.evidence),
                SEP,
            ]
        for line in lines:
            print(line)
        print()

    # -- Step 4: Closing summary --
    print(
        f"{total} of {total} failures handled gracefully"
        " — 0 crashes,"
    )
    print(
        "0 silent errors, 0 guessed matches. Every"
        " decision is in"
    )
    print("the audit trail.")

    # -- Step 5: Log to demo audit file --
    log_demo_failures(failures)
    print(
        f"\nDemo audit log written: {DEMO_AUDIT_PATH}"
        f" ({len(failures)} events)"
    )
    print()


if __name__ == "__main__":
    main()

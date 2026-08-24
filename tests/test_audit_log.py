"""Focused tests for the append-only audit log.

Runs with no dependencies:
    python tests/test_audit_log.py

and is picked up unchanged by pytest if it is ever installed:
    pytest tests/test_audit_log.py

The tests concentrate on the four properties that make this log worth having,
because everything else about it is ordinary file I/O:

  1. Appending never modifies a byte that was already written.
  2. Reads come back in chronological (append) order, and that order does not
     depend on wall-clock timestamps.
  3. Money is integer paise end to end — rejected at construction, and still
     an integer in the raw JSON on disk.
  4. delta == actual - expected, always.

A test that merely round-trips an object through the API would pass even if the
implementation rewrote the file on every append, so several tests assert against
the raw bytes rather than through the reader.
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit_log import (  # noqa: E402
    MONEY_FIELDS, SCHEMA_VERSION, AuditEvent, AuditLog, AuditLogCorruption,
    AuditLogError, MatchMethod, MatchStatus,
)
from settlement_math import (  # noqa: E402
    compute_payment, reconcile, roll_up, to_paise,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tmp_log(**kw) -> AuditLog:
    d = tempfile.mkdtemp(prefix="settlesense_audit_")
    # fsync off in tests: we assert on file contents, not on power-loss survival,
    # and a syscall per append makes the suite needlessly slow.
    return AuditLog(Path(d) / "audit.jsonl", fsync=False, **kw)


def _event(bank_id="bank_0", expected=100_000, actual=100_000, **kw) -> AuditEvent:
    """A valid event with sensible defaults, so each test states only what it cares about."""
    kw.setdefault("status", MatchStatus.MATCHED)
    kw.setdefault("match_method", MatchMethod.UTR_EXACT)
    kw.setdefault("confidence", 1.0)
    return AuditEvent.create(
        bank_record_id=bank_id,
        expected_amount_paise=expected,
        actual_amount_paise=actual,
        **kw,
    )


def _raises(exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from e
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


def _concurrent_worker(path: str, worker: int, count: int) -> None:
    """Append `count` events to `path`. Module-level so it survives both the
    fork and spawn multiprocessing start methods."""
    log = AuditLog(path, fsync=False)
    for i in range(count):
        log.append(AuditEvent.create(
            bank_record_id=f"bank_w{worker}_{i}",
            status=MatchStatus.MATCHED,
            match_method=MatchMethod.UTR_EXACT,
            expected_amount_paise=1000 + i,
            actual_amount_paise=1000 + i,
            confidence=1.0,
            reason=f"worker {worker} event {i}",
        ))


# ---------------------------------------------------------------------------
# 1. append-only / immutability
# ---------------------------------------------------------------------------

def test_append_never_rewrites_earlier_bytes():
    """The core guarantee, asserted at the byte level.

    Snapshot the whole file after each append and require every snapshot to be a
    strict byte prefix of the next. Any in-place edit, reordering or reformat of
    an earlier event breaks the prefix property, which a read-through assertion
    would happily miss.
    """
    log = _tmp_log()
    snapshots = []
    for i in range(6):
        log.append(_event(f"bank_{i}", expected=1000 * i, actual=1000 * i))
        snapshots.append(log.path.read_bytes())

    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier), (
            "an append modified previously written bytes: "
            f"{earlier[-80:]!r} is not a prefix of {later[-80:]!r}"
        )
    assert len(snapshots[-1]) > len(snapshots[0]), "file never grew"


def test_appending_leaves_first_event_identical():
    """Re-reading event 0 after later appends must yield exactly the same record."""
    log = _tmp_log()
    first = log.append(_event("bank_0", expected=4_783_200, actual=4_783_200))
    line0_before = log.path.read_text(encoding="utf-8").splitlines()[0]

    for i in range(1, 4):
        log.append(_event(f"bank_{i}", expected=500, actual=400))

    line0_after = log.path.read_text(encoding="utf-8").splitlines()[0]
    assert line0_before == line0_after, "first line changed after later appends"
    assert log.read_events()[0].to_dict() == first.to_dict()


def test_log_exposes_no_mutating_api():
    """Absence of update/delete is the feature; guard it against a future 'helpful' patch."""
    for forbidden in ("update", "delete", "remove", "truncate", "clear",
                      "overwrite", "rewrite", "edit", "pop", "insert"):
        assert not hasattr(AuditLog, forbidden), (
            f"AuditLog grew a mutating method {forbidden!r} — this breaks the "
            f"append-only guarantee"
        )


def test_events_are_frozen():
    """An event already handed to the log cannot be edited in memory either."""
    ev = _event()
    _raises(Exception, setattr, ev, "delta_paise", 999)
    _raises(Exception, setattr, ev, "status", MatchStatus.EXCEPTION)


def test_append_does_not_mutate_callers_event():
    """`seq` stamping must return a new object, not edit the caller's."""
    ev = _event()
    assert ev.seq == -1
    log = _tmp_log()
    stamped = log.append(ev)
    assert ev.seq == -1, "append mutated the caller's event"
    assert stamped.seq == 0
    assert stamped is not ev


def test_reopening_continues_seq_instead_of_colliding():
    """A fresh process must not restart numbering and duplicate seq values."""
    log = _tmp_log()
    log.append_many([_event("bank_0"), _event("bank_1")])

    reopened = AuditLog(log.path, fsync=False)
    reopened.append(_event("bank_2"))

    seqs = [e.seq for e in reopened.read_events()]
    assert seqs == [0, 1, 2], f"seq restarted or collided: {seqs}"
    assert len(set(seqs)) == len(seqs)


# ---------------------------------------------------------------------------
# 2. chronological ordering
# ---------------------------------------------------------------------------

def test_concurrent_writers_lose_nothing_and_do_not_collide_seq():
    """Eight processes appending to one log: the claim most likely to be false.

    Two properties matter. Nothing may be lost or torn (O_APPEND gives us that),
    and `seq` must stay unique — assigning it is a read-then-write race, and
    before the sidecar lock this produced 229 distinct seq values out of 400
    events. A ledger whose sequence number silently repeats is worse than one
    with no sequence number, because someone will use it as a key.
    """
    log = _tmp_log()
    workers, per_worker = 8, 25
    procs = [
        mp.Process(target=_concurrent_worker, args=(str(log.path), w, per_worker))
        for w in range(workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs), \
        f"a writer process failed: {[p.exitcode for p in procs]}"

    expected_total = workers * per_worker
    events = log.read_events()          # raises if any line is torn
    assert len(events) == expected_total, \
        f"lost events: {len(events)} of {expected_total}"

    bank_ids = {e.bank_record_id for e in events}
    assert len(bank_ids) == expected_total, "an event was overwritten"

    seqs = [e.seq for e in events]
    assert len(set(seqs)) == expected_total, (
        f"seq collided under concurrency: {expected_total - len(set(seqs))} "
        f"duplicate(s)"
    )
    assert sorted(seqs) == list(range(expected_total)), \
        f"seq is not a dense monotonic range: {sorted(seqs)[:10]}..."


def test_read_events_is_chronological():
    log = _tmp_log()
    ids = [f"bank_{i}" for i in range(10)]
    for bid in ids:
        log.append(_event(bid))
    assert [e.bank_record_id for e in log.read_events()] == ids
    assert [e.seq for e in log.read_events()] == list(range(10))


def test_order_survives_a_clock_that_steps_backwards():
    """Ordering is append order, not timestamp order.

    NTP corrections and same-microsecond appends must not be able to reorder the
    record of what happened, so the reader must not sort on `timestamp`.
    """
    log = _tmp_log()
    log.append(_event("bank_first", timestamp="2026-08-23T10:00:00+00:00"))
    log.append(_event("bank_second", timestamp="2020-01-01T00:00:00+00:00"))
    log.append(_event("bank_third", timestamp="2026-08-23T10:00:00+00:00"))

    got = [e.bank_record_id for e in log.read_events()]
    assert got == ["bank_first", "bank_second", "bank_third"], (
        f"reader reordered events by timestamp: {got}"
    )


def test_append_many_preserves_order_and_is_equivalent_to_singles():
    a, b = _tmp_log(), _tmp_log()
    evs = [_event(f"bank_{i}", expected=i * 7, actual=i * 7) for i in range(5)]
    a.append_many(evs)
    for ev in evs:
        b.append(ev)
    assert [e.bank_record_id for e in a.read_events()] == \
           [e.bank_record_id for e in b.read_events()]
    assert [e.seq for e in a.read_events()] == [0, 1, 2, 3, 4]


def test_iter_events_matches_read_events():
    log = _tmp_log()
    log.append_many([_event(f"bank_{i}") for i in range(4)])
    assert [e.seq for e in log.iter_events()] == [e.seq for e in log.read_events()]


def test_empty_and_missing_log_read_cleanly():
    log = _tmp_log()
    assert log.read_events() == [] and log.count() == 0 and len(log) == 0
    log.path.write_text("", encoding="utf-8")
    assert log.read_events() == []
    assert log.append_many([]) == [], "empty batch should be a no-op"
    assert not log.path.read_bytes(), "empty batch created content"


# ---------------------------------------------------------------------------
# 3. money is integer paise, never float
# ---------------------------------------------------------------------------

def test_float_amounts_are_rejected_not_coerced():
    """Rounding a float silently would put a wrong number in a permanent record."""
    for bad in (478.32, 100000.0, -0.5):
        _raises(AuditLogError, _event, expected=bad)
        _raises(AuditLogError, _event, actual=bad)


def test_bool_is_not_an_amount():
    """bool is an int subclass in Python; True is not 1 paisa."""
    _raises(AuditLogError, _event, expected=True)
    _raises(AuditLogError, _event, actual=False)


def test_decimal_and_string_amounts_are_rejected():
    from decimal import Decimal
    _raises(AuditLogError, _event, expected=Decimal("478.32"))
    _raises(AuditLogError, _event, expected="478.32")
    _raises(AuditLogError, _event, expected=None)


def test_money_serialises_as_json_integers_on_disk():
    """Assert on the raw text: `1000` must not have become `1000.0`."""
    log = _tmp_log()
    log.append(_event(expected=4_783_200, actual=4_783_100))
    raw = log.path.read_text(encoding="utf-8").strip()

    parsed = json.loads(raw)
    for f in MONEY_FIELDS:
        assert isinstance(parsed[f], int) and not isinstance(parsed[f], bool), (
            f"{f} serialised as {type(parsed[f]).__name__}: {parsed[f]!r}"
        )
        # Assert on the literal text too: json.loads would happily turn
        # "4783200.0" back into a float-valued key that passes isinstance(int)
        # only because Python coerced it. The bytes must show no decimal point.
        assert f'"{f}": {parsed[f]}' in raw, (
            f"{f} is not written as a bare integer literal in {raw}"
        )
    assert "NaN" not in raw and "Infinity" not in raw


def test_large_and_negative_amounts_survive_exactly():
    """Refunds are negative and crores are big; neither may lose precision."""
    log = _tmp_log()
    big = 99_99_99_99_999           # ~₹10 crore in paise
    log.append(_event("bank_refund", expected=-1_957_52, actual=-1_957_52))
    log.append(_event("bank_big", expected=big, actual=big - 1))

    events = log.read_events()
    assert events[0].expected_amount_paise == -195752
    assert events[1].expected_amount_paise == big
    assert events[1].delta_paise == -1


def test_metadata_paise_keys_obey_the_integer_rule():
    """The free-form escape hatch must not become the hole floats leak through."""
    _raises(AuditLogError, _event, metadata={"tolerance_paise": 1.5})
    ok = _event(metadata={"tolerance_paise": 100})
    assert ok.metadata["tolerance_paise"] == 100


def test_non_serialisable_metadata_fails_before_writing():
    """Validation must happen at construction, not halfway through an append."""
    _raises(AuditLogError, _event, metadata={"obj": object()})
    _raises(AuditLogError, _event, metadata={"nan": float("nan")})
    _raises(AuditLogError, _event, metadata={"bad_key": {1, 2}})


def test_nan_confidence_is_rejected():
    """NaN is not legal JSON and would poison every downstream reader."""
    _raises(AuditLogError, _event, confidence=float("nan"))
    _raises(AuditLogError, _event, confidence=float("inf"))
    _raises(AuditLogError, _event, confidence=1.5)
    _raises(AuditLogError, _event, confidence=-0.1)


# ---------------------------------------------------------------------------
# 4. delta consistency
# ---------------------------------------------------------------------------

def test_delta_is_derived_and_signed():
    """Sign matters: over-credit and short-pay are different investigations."""
    assert _event(expected=100_000, actual=100_000).delta_paise == 0
    assert _event(expected=100_000, actual=99_900).delta_paise == -100
    assert _event(expected=100_000, actual=100_100).delta_paise == 100


def test_contradictory_explicit_delta_is_rejected():
    _raises(
        AuditLogError, _event,
        expected=100_000, actual=99_900, delta_paise=0,
    )
    consistent = _event(expected=100_000, actual=99_900, delta_paise=-100)
    assert consistent.delta_paise == -100


# ---------------------------------------------------------------------------
# field coverage, round-trip, vocabulary
# ---------------------------------------------------------------------------

def test_all_required_fields_round_trip():
    log = _tmp_log()
    ev = AuditEvent.create(
        bank_record_id="bank_7",
        status=MatchStatus.EXCEPTION,
        match_method=MatchMethod.AMOUNT_WINDOW,
        expected_amount_paise=4_783_200,
        actual_amount_paise=4_783_100,
        settlement_ids=["setl_004", "setl_005"],
        payment_ids=["pay_012", "pay_013"],
        candidate_ids=["setl_004", "setl_009", "setl_011"],
        confidence=0.62,
        reason="two batches share this amount — undecidable on amount alone",
        metadata={"trap_name": "ambiguous_amount", "tolerance_paise": 100},
    )
    log.append(ev)
    got = log.read_events()[0]

    assert got.event_id == ev.event_id and got.event_id.startswith("evt_")
    assert got.timestamp == ev.timestamp
    assert got.bank_record_id == "bank_7"
    assert got.settlement_ids == ("setl_004", "setl_005")
    assert got.payment_ids == ("pay_012", "pay_013")
    assert got.status is MatchStatus.EXCEPTION
    assert got.match_method is MatchMethod.AMOUNT_WINDOW
    assert got.candidate_ids == ("setl_004", "setl_009", "setl_011")
    assert got.expected_amount_paise == 4_783_200
    assert got.actual_amount_paise == 4_783_100
    assert got.delta_paise == -100
    assert abs(got.confidence - 0.62) < 1e-9
    assert "undecidable" in got.reason
    assert got.metadata["trap_name"] == "ambiguous_amount"
    assert got.schema_version == SCHEMA_VERSION
    assert got.seq == 0


def test_event_ids_are_unique():
    ids = {_event().event_id for _ in range(500)}
    assert len(ids) == 500, "event_id collided"


def test_metadata_is_optional():
    ev = _event()
    assert ev.metadata == {}
    log = _tmp_log()
    log.append(ev)
    assert log.read_events()[0].metadata == {}


def test_unknown_status_or_method_is_rejected():
    _raises(AuditLogError, _event, status="probably_fine")
    _raises(AuditLogError, _event, match_method="vibes")


def test_status_and_method_vocabulary_matches_ground_truth_columns():
    """These strings are compared directly against generate_data.py's
    `expected_outcome` / `expected_method` columns, so drift here silently breaks
    any future comparison against ground truth."""
    assert {s.value for s in MatchStatus} >= {"matched", "exception"}
    assert {m.value for m in MatchMethod} >= {
        "utr_exact", "batch_by_utr", "amount_window", "none"
    }


def test_bad_id_lists_are_rejected():
    _raises(AuditLogError, _event, settlement_ids="setl_004")   # bare string
    _raises(AuditLogError, _event, payment_ids=["pay_1", ""])   # blank
    _raises(AuditLogError, _event, candidate_ids=[None])
    _raises(AuditLogError, _event, bank_id="")
    _raises(AuditLogError, _event, bank_id="   ")


def test_unicode_narration_stays_on_one_physical_line():
    """`fmt()` emits ₹, and narrations carry it. One event must be one line."""
    log = _tmp_log()
    log.append(_event(reason="short-paid by ₹1,957.52 — investigate  ✓"))
    log.append(_event(reason="second event"))

    text = log.path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2, "an event spilled across lines"
    assert "₹1,957.52" in log.read_events()[0].reason


def test_only_audit_events_can_be_appended():
    log = _tmp_log()
    _raises(AuditLogError, log.append, {"bank_record_id": "bank_0"})
    _raises(AuditLogError, log.append, "not an event")
    assert log.count() == 0, "a rejected append still wrote to the file"


# ---------------------------------------------------------------------------
# corruption is a finding, not noise
# ---------------------------------------------------------------------------

def test_metadata_paise_check_recurses_into_nested_structures():
    """A flat-only check would pass the very shape callers are most likely to use.

    `{"breakdown": {"fee_paise": 1.5}}` is how you would record a per-payment
    breakdown, so if the `_paise` convention is not enforced at depth it is not
    really enforced.
    """
    _raises(AuditLogError, _event, metadata={"breakdown": {"fee_paise": 1.5}})
    _raises(AuditLogError, _event, metadata={"rows": [{"fee_paise": 1.5}]})
    _raises(AuditLogError, _event,
            metadata={"a": {"b": {"c": [{"net_paise": 2.5}]}}})

    # ints at depth are fine, and floats under non-money keys are left alone:
    # metadata legitimately carries scores and durations.
    ok = _event(metadata={"breakdown": {"fee_paise": 150}, "elapsed_ms": 12.5})
    assert ok.metadata["breakdown"]["fee_paise"] == 150


def test_read_path_enforces_the_same_invariants_as_the_write_path():
    """A reader that rejects a bad status but shrugs at confidence=99 is only
    pretending to validate. Asymmetry here is how bad records become permanent."""
    def _tampered(field, value):
        log = _tmp_log()
        log.append(_event("bank_0"))
        d = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
        d[field] = value
        log.path.write_text(json.dumps(d) + "\n", encoding="utf-8")
        return log

    for field, value in (
        ("confidence", 99.0),        # create() enforces [0, 1]
        ("confidence", -5),
        ("bank_record_id", ""),      # create() requires non-empty
        ("bank_record_id", "   "),
        ("status", "bogus"),
        ("match_method", "vibes"),
        ("metadata", {"tolerance_paise": 1.5}),
    ):
        _raises(AuditLogCorruption, _tampered(field, value).read_events)


def test_tampered_delta_on_disk_is_detected_on_read():
    """The reader must re-check the delta identity, not just the writer.

    `create()` refuses a contradictory delta, but a line can reach the file from
    a hand edit or another tool. Zeroing `delta_paise` is exactly how you would
    hide a shortfall in a ledger whose own operands still prove it, so the read
    path has to catch it — otherwise the log validates its input but not its
    contents.
    """
    log = _tmp_log()
    log.append(_event("bank_0", expected=100_000, actual=95_000))
    d = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    assert d["delta_paise"] == -5_000
    d["delta_paise"] = 0                      # hide the ₹50 shortfall
    log.path.write_text(json.dumps(d) + "\n", encoding="utf-8")

    _raises(AuditLogCorruption, log.read_events)


def test_tampered_amount_on_disk_is_detected_on_read():
    """Editing an amount without fixing the delta is equally detectable."""
    log = _tmp_log()
    log.append(_event("bank_0", expected=100_000, actual=100_000))
    d = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    d["actual_amount_paise"] = 1              # delta still says 0
    log.path.write_text(json.dumps(d) + "\n", encoding="utf-8")

    _raises(AuditLogCorruption, log.read_events)


def test_corrupt_line_raises_with_line_number():
    log = _tmp_log()
    log.append(_event("bank_0"))
    with log.path.open("a", encoding="utf-8") as f:
        f.write("{not json at all}\n")
    log.append(_event("bank_2"))

    try:
        log.read_events()
        raise AssertionError("corrupt line was silently accepted")
    except AuditLogCorruption as exc:
        assert exc.line_number == 2, f"wrong line reported: {exc.line_number}"


def test_missing_required_field_is_corruption():
    log = _tmp_log()
    with log.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "evt_x", "timestamp": "now"}) + "\n")
    _raises(AuditLogCorruption, log.read_events)


def test_float_money_on_disk_is_corruption():
    """Defends the integer rule against a record written by anything but this module."""
    log = _tmp_log()
    log.append(_event())
    good = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    good["expected_amount_paise"] = 4783.32
    with log.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n")
    _raises(AuditLogCorruption, log.read_events)


def test_future_schema_version_refuses_to_be_guessed_at():
    log = _tmp_log()
    log.append(_event())
    d = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    d["schema_version"] = SCHEMA_VERSION + 1
    with log.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")
    _raises(AuditLogCorruption, log.read_events)


def test_torn_final_line_can_be_tolerated_but_not_by_default():
    """The one failure a crash mid-append can actually cause."""
    log = _tmp_log()
    log.append_many([_event("bank_0"), _event("bank_1")])
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"event_id": "evt_trunc", "bank_record')  # no newline

    _raises(AuditLogCorruption, log.read_events)
    survivors = log.read_events(tolerate_partial_tail=True)
    assert [e.bank_record_id for e in survivors] == ["bank_0", "bank_1"], (
        "tolerating a torn tail lost committed events"
    )


def test_blank_lines_are_ignored():
    log = _tmp_log()
    log.append(_event("bank_0"))
    with log.path.open("a", encoding="utf-8") as f:
        f.write("\n   \n")
    log.append(_event("bank_1"))
    assert [e.bank_record_id for e in log.read_events()] == ["bank_0", "bank_1"]


# ---------------------------------------------------------------------------
# integration with settlement_math (the real reason this schema looks like it does)
# ---------------------------------------------------------------------------

def test_from_reconciliation_records_an_exact_match():
    batch = roll_up(
        [compute_payment(to_paise("15000.00")), compute_payment(to_paise("22500.00"))],
        ["pay_012", "pay_013"],
    )
    rec = reconcile(batch.net, batch)

    log = _tmp_log()
    log.append(AuditEvent.from_reconciliation(
        "bank_7", rec, match_method=MatchMethod.BATCH_BY_UTR,
        settlement_ids=["setl_004"], confidence=1.0,
    ))
    ev = log.read_events()[0]

    assert ev.status is MatchStatus.MATCHED
    assert ev.delta_paise == 0
    assert ev.expected_amount_paise == batch.net
    assert ev.actual_amount_paise == batch.net
    assert ev.payment_ids == ("pay_012", "pay_013")
    assert ev.metadata["verdict"] == "exact"
    assert isinstance(ev.metadata["tolerance_paise"], int)
    assert "expected net" in ev.reason


def test_from_reconciliation_records_a_variance_as_an_exception():
    batch = roll_up([compute_payment(to_paise("15000.00"))], ["pay_020"])
    rec = reconcile(batch.net - 5000, batch)          # ₹50 short — beyond tolerance

    ev = AuditEvent.from_reconciliation("bank_9", rec,
                                        match_method=MatchMethod.AMOUNT_WINDOW)
    assert ev.status is MatchStatus.EXCEPTION
    assert ev.delta_paise == -5000
    assert ev.metadata["verdict"] == "variance"


def test_from_reconciliation_within_tolerance_is_matched_but_traceable():
    """Tolerance hides money, so the log must record that it was used."""
    batch = roll_up([compute_payment(to_paise("15000.00"))], ["pay_021"])
    rec = reconcile(batch.net - 50, batch)            # 50 paise short

    ev = AuditEvent.from_reconciliation("bank_10", rec,
                                       match_method=MatchMethod.UTR_EXACT)
    assert ev.status is MatchStatus.MATCHED
    assert ev.delta_paise == -50
    assert ev.metadata["verdict"] == "within_tolerance", (
        "a tolerance-assisted match is indistinguishable from an exact one"
    )


def test_from_reconciliation_carries_diagnostic_notes():
    """The 'fees not deducted' note is the most valuable line reconcile() emits."""
    batch = roll_up([compute_payment(to_paise("15000.00"))], ["pay_022"])
    rec = reconcile(batch.gross, batch)               # credit == gross

    ev = AuditEvent.from_reconciliation("bank_11", rec)
    assert ev.status is MatchStatus.EXCEPTION
    assert any("GROSS" in n for n in ev.metadata["notes"])


def test_from_reconciliation_rejects_a_wrong_shaped_object():
    _raises(AuditLogError, AuditEvent.from_reconciliation, "bank_0", object())


def test_reconciliation_amounts_are_never_floats():
    """End-to-end: money from settlement_math through the log stays integral."""
    batch = roll_up([compute_payment(to_paise("15000.00"))], ["pay_030"])
    rec = reconcile(batch.net - 1, batch)
    log = _tmp_log()
    log.append(AuditEvent.from_reconciliation("bank_12", rec))

    parsed = json.loads(log.path.read_text(encoding="utf-8").strip())
    for f in MONEY_FIELDS:
        assert isinstance(parsed[f], int), f"{f} is {type(parsed[f]).__name__}"
    assert parsed["delta_paise"] == -1


# ---------------------------------------------------------------------------
# runner (so this file works without pytest, per repo convention)
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed, failures = 0, []
    print("=" * 78)
    print(f"audit log — {len(tests)} focused tests")
    print("=" * 78)
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
        else:
            passed += 1
            print(f"  ok    {name}")
    print("-" * 78)
    if failures:
        print(f"FAILED — {len(failures)} of {len(tests)}")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print(f"ALL {passed} AUDIT LOG TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

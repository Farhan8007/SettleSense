![Tests](https://img.shields.io/badge/tests-350%2F350%20passing-brightgreen)
![Accuracy](https://img.shields.io/badge/accuracy-100%25-brightgreen)
![False Positives](https://img.shields.io/badge/false%20positives-0-brightgreen)

# SettleSense — AI Finance Controller & Reconciliation Engine

> **Razorpay AI Buildathon — Track 4: AI Finance Controller**  
> *Deterministic 3-Pass Matching, Typed Exception Triage, and Tamper-Evident Audit Logging for Automated Payment Gateway Reconciliation.*

---

## 1. Overview

SettleSense is an automated financial reconciliation engine designed to bridge the gap between bank statement credits, payment gateway settlement batches (Razorpay), and internal merchant order ledgers. 

Reconciliation in real-world finance is complex. Payment gateways do not pay out transactions individually; they aggregate hundreds of customer payments into net settlement batches, deduct processing fees (1.8% + 18% GST), apply adjustments or refunds, and transfer a single net amount to the bank. Bank statements frequently truncate transaction descriptions, strip UTR numbers, or display cryptic narration strings. SettleSense resolves these discrepancies using a deterministic 3-pass matching engine, routes unmatched items to a typed exception triage queue with rich diagnostic evidence, and logs every state transition to an append-only, tamper-evident audit trail.

---

## Buildathon Requirements — At a Glance

| Requirement | How SettleSense satisfies it |
|---|---|
| **Explainable** | Every match/exception decision has a human-readable reason + full evidence — see the audit trail ([`dashboard/dashboard.html`](dashboard/dashboard.html) → Audit Trail tab, or [`out/audit_exceptions.jsonl`](out/audit_exceptions.jsonl)) |
| **Bounded** | Hard caps enforced in code — e.g. max 20 AI calls per run in Pass 3, one-to-one batch consumption ([`src/matcher.py`](src/matcher.py#L718)) |
| **Gated** | Deterministic passes (exact match, batch-sum) always run BEFORE any AI call; AI is last-resort only, never trusted blindly — see [`src/matcher.py`](src/matcher.py#L718) Pass 3 |
| **Audit trail** | Append-only, tamper-evident JSONL log — corrections are new events, never edits ([`src/audit_log.py`](src/audit_log.py)) |
| **One failure handled gracefully** | [`demo/failure_demo.py`](demo/failure_demo.py) — narrates 3 real failure types live: duplicate credit, variance breach, ambiguous amount |
| **Throughput + measured accuracy** | 100.0% accuracy, 39/39 correct against pre-committed ground truth ([`tests/accuracy_report.py`](tests/accuracy_report.py)) |
| **Honest exception list (not cherry-picked)** | 13 exceptions shown with real evidence, including the ones we got right AND the full false-positive/negative count (0/0) |
| **AI Judgment (deterministic where AI unneeded)** | Only 1 of 33 bank credits needs AI at all — Pass 1+2 resolve the rest deterministically |
| **Failure Recovery** | A real bug was found (Pass 2 double-exception issue), root-caused, fixed, and the fix is documented with before/after proof — see [`ARCHITECTURE.md` Section 5](ARCHITECTURE.md#5-developer-retrospective-the-pass-2-state-pipeline-bug) |

---

## 2. Quickstart

Run these exact commands from the project root to reproduce the full pipeline, generate reports, launch the interactive dashboard, and run the failure demonstration:

```bash
# 1. Generate synthetic dataset with pre-committed ground truth
python3 src/generate_data.py

# 2. Run 3-pass matching engine, exception classifier, and build dashboard payload
python3 dashboard/build_dashboard_data.py

# 3. View the standalone interactive dashboard (Open in browser)
open dashboard/dashboard.html

# 4. Run the live failure handling demo (narrates 3 real edge cases live)
python3 demo/failure_demo.py

# 5. Run ground-truth accuracy report
python3 tests/accuracy_report.py

# 6. Run dataset integrity checks & audit log test suites
python3 tests/check_dataset.py
python3 tests/test_audit_log.py
python3 tests/verify_exceptions.py

# 7. (Optional) Run Razorpay API test-mode touchpoint integration
python3 demo/razorpay_touchpoint_demo.py
```

---

## 3. Results & Ground-Truth Verification

Evaluated against a pre-committed ground-truth dataset generated with fixed seed `20260823`:

| Metric | Result | Context / Breakdown |
| :--- | :---: | :--- |
| **Overall Accuracy** | **100.0%** | **39/39** correct decisions on ground truth (0 false positives, 0 false negatives) |
| **Bank Credit Match Rate** | **78.8%** | **26 / 33** bank credits correctly matched to settlement batches |
| **Bank-Side Exceptions** | **7** | 7 bank credits correctly identified as exceptions |
| **Total Classified Exceptions** | **13** | 7 bank-side + 3 `missing_settlement` + 3 `order_never_settled` |
| **Dataset Integrity Checks** | **241 / 241** | 100% pass rate across mathematical identity, UTR mapping, and fee rules |
| **Audit Log Unit Tests** | **46 / 46** | 100% pass rate on immutability, sequence integrity, and error recovery |

> **Note on Accuracy Framing:** The 100.0% accuracy metric reflects complete evaluation against our forward-generated ground-truth dataset containing 12 engineered trap types. In production, unmodeled bank narration formats or systemic gateway changes require ongoing rule refinement.

---

## 4. System Architecture

```
                       ┌─────────────────────────────────────────┐
                       │               DATA SOURCES              │
                       │  Bank Statement CSV  │  Razorpay CSV    │
                       │  Order Ledger CSV    │  Razorpay API    │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │            DATA NORMALIZER              │
                       │  • Integer Paise Conversion (No Floats) │
                       │  • Regex UTR & Reference Extraction     │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │         3-PASS MATCHING ENGINE          │
                       │                                         │
                       │  Pass 1: Deterministic Exact UTR Match  │
                       │     │ (Unmatched)                       │
                       │     ▼                                   │
                       │  Pass 2: Batch Net Sum & Window Match   │
                       │          (≤100 Paise Tolerance)         │
                       │     │ (Unmatched)                       │
                       │     ▼                                   │
                       │  Pass 3: AI Narration Extraction (Groq) │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │      EXCEPTION CLASSIFIER & AUDIT       │
                       │  • Typed Exceptions & Severity          │
                       │  • Immutable JSONL Audit Event Stream   │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │              OUTPUT LAYER               │
                       │  • Single-File dashboard.html           │
                       │  • CLI Accuracy & Verification Reports  │
                       └─────────────────────────────────────────┘
```

---

## 5. The 12 Engineered Edge Case Traps

The test dataset incorporates 12 specific financial and operational edge cases to evaluate engine resilience:

1. **`clean_single`**: Standard 1:1 match between bank statement credit and settlement batch.
2. **`clean_batch`**: Standard 1:N batch where multiple customer payments sum to a single net payout.
3. **`batch_no_utr`**: Settlement batch missing UTR in gateway export, resolved via net amount sum within the date window.
4. **`duplicate_credit`**: Duplicate UTR deposit appearing twice in bank statement; second occurrence flagged as duplicate to prevent revenue double-counting.
5. **`short_pay_within_tolerance`**: Bank credit off by ≤100 paise (e.g. ₹0.50 bank clearance fee); matched within declared tolerance.
6. **`variance_breach`**: Net payout discrepancy > 100 paise (e.g. unannounced gateway fee change); flagged as variance breach exception with exact delta and diagnosed cause.
7. **`fees_not_deducted`**: Bank credited gross settlement amount instead of net payout after fees; flagged as variance breach.
8. **`zero_value_settlement`**: Net settlement payout is exactly ₹0.00 due to fee/refund offsets; handled cleanly without division-by-zero or matching failure.
9. **`refund_batch`**: Net settlement payout is negative due to net refund volume exceeding sales; matched by UTR and negative net sum.
10. **`ambiguous_amount`**: Multiple settlement batches match the exact net credit amount within the date window; routed to human review rather than making an ungrounded guess.
11. **`unparseable_narration_recovered` & `unparseable_narration_unresolvable`**: Truncated/scrambled bank narration strings; either recovered via date+amount windowing or escalated to AI/human queue.
12. **`orphan_bank_credit` & `missing_settlement` & `order_never_settled`**: Unrecorded deposits, missing gateway batches, or merchant orders stuck in pending/failed status without settlement.

---

## 6. Tech Stack

- **Python 3.10+**: Core engine logic, matching algorithms, exception classification, and CLI tools using standard library + `pandas` & `python-dotenv`.
- **Groq API (`qwen/qwen3.6-27b`, `reasoning_effort="none"`)**: Fast, ultra-low-latency LLM inference used strictly in Pass 3 for unparseable bank narration text extraction.
- **Vanilla HTML5 / CSS3 / JavaScript (`dashboard/dashboard.html`)**: Zero-build-step, standalone client-side dashboard with interactive filtering, search, and audit trail rendering.
- **Razorpay API Python SDK**: Fetching real test-mode payments and seeding test payment links (`demo/create_test_payment_links.py`).
- **JSONL Audit Storage**: Append-only, schema-validated, line-delimited JSON files providing immutable audit logging.

---

## 7. Deep Architecture & Design Rationale

For an in-depth breakdown of mathematical design choices, integer paise representation, forward dataset generation, the Pass 2 developer retrospective, and production trade-offs, see [ARCHITECTURE.md](ARCHITECTURE.md).

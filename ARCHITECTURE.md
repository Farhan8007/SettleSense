# SettleSense — Architecture & Design Rationale

> **Technical Whitepaper & System Architecture**  
> *Detailed design rationale, data integrity guarantees, engine mechanics, developer retrospective, and production readiness.*

---

## 1. Forward-Generation Architecture: Ground Truth by Construction

A core engineering decision in SettleSense is generating evaluation datasets through **forward construction** rather than hand-writing synthetic bank CSVs or labeling records manually after the fact.

### How Forward-Generation Works
1. **Order Creation**: Orders are initialized in the merchant ledger with fixed IDs, timestamps, and gross amounts.
2. **Payment Processing**: Payments are generated against orders, applying Razorpay's standard fee structure (1.8% fee + 18% GST on fee = 2.124% total deduction).
3. **Settlement Batching**: Captured payments are grouped into daily/periodic settlement batches, computing the exact net payout sum.
4. **Bank Statement Deposit**: Bank credits are generated from settlement batches, introducing realistic operational noise (truncated narrations, missing UTRs, duplicate transfers, date drift, fee rounding).

```
  [ Merchant Orders ] ──► [ Razorpay Payments ] ──► [ Settlement Batches ] ──► [ Bank Credits ]
                                                                                   │
  [ GROUND TRUTH MATRIX ] ◄────────────────────────────────────────────────────────┘
  (Exact expected matches, delta paise, and exception types produced as a direct byproduct)
```

### Benefits
- **Zero Labeling Bias**: Ground truth is a mathematical byproduct of generation using seed `20260823`, ensuring 100% reproducible test data.
- **Exact Accounting Identity**: Every test record guarantees that `Net Payout = Gross - Fee - GST` down to the single paise across all 241 integrity checks (`tests/check_dataset.py`).

---

## 2. Integer Paise Domain Logic: Zero-Float Financial Execution

In financial software, using binary floating-point representation (`float` or `double`) introduces IEEE 754 rounding artifacts (e.g., `0.1 + 0.2 = 0.30000000000000004` or `₹1,000.05 * 0.018 = ₹18.0009`). Accumulated over thousands of transactions, floating-point drift creates artificial reconciliation discrepancies.

### Enforced Rules
- **Signed 64-Bit Integers**: All monetary amounts are converted to integer paise (1 INR = 100 paise, e.g., ₹239.79 → `23979` paise) at the ingestion boundary (`src/normalizer.py`).
- **Disk & API Representation**: Money fields saved to JSON, JSONL, or passed via internal structures use integer values ending in `_paise` (e.g., `amount_paise`, `delta_paise`).
- **Formatting Boundary**: Conversion to formatted currency strings (`₹239.79`) occurs strictly at the presentation layer (`src/settlement_math.py`'s `fmt()` helper and `dashboard.html`).

---

## 3. The 3-Pass Reconciliation Engine

SettleSense organizes matching into three distinct passes ordered strictly by **precision, performance, and operational cost**:

```
                    ┌────────────────────────────────────────┐
                    │       Bank Credit & Batch Input        │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │ PASS 1: Exact UTR Match (O(1) Hash Lookup)                           │
    │ • Matches Bank UTR directly against Gateway Settlement UTR            │
    │ • High precision, 0% hallucination risk                               │
    └───────────────────────────────────┬───────────────────────────────────┘
                                        │ (Unmatched)
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │ PASS 2: Batch Net Sum & Window Match (Deterministic Heuristics)       │
    │ • Groups unmatched credits by net amount within ±3-day window         │
    │ • Allows declared tolerance (≤100 paise) for minor bank fee rounding  │
    │ • UTR-prioritization narrows ambiguous candidate batches              │
    └───────────────────────────────────┬───────────────────────────────────┘
                                        │ (Unmatched & Unparseable)
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │ PASS 3: AI-Assisted Narration Parser (Groq qwen/qwen3.6-27b)          │
    │ • Invoked ONLY for scrambled/truncated narrations (e.g., bank_31)     │
    │ • Extracts UTRs/Ref numbers from unstructured text                    │
    │ • Extraction candidate MUST validate against settlement index         │
    └───────────────────────────────────────────────────────────────────────┘
```

### Why AI is Restricted to Pass 3
- **Determinism First**: Over 95% of standard settlement records resolve deterministically in Passes 1 and 2.
- **Constrained AI Output**: The LLM is never allowed to "decide" a match or alter financial balances. It functions purely as an entity extractor. Any extracted UTR or reference number is strictly re-validated against the settlement index before accepting a match.

---

## 4. Append-Only Audit Logging Architecture

Financial compliance requires complete auditability. Modifying or deleting audit records on disk violates accounting controls.

### Implementation
- **Immutable Log Stream**: Audit events are persisted to line-delimited JSON (`.jsonl`) files.
- **State Updates as New Events**: Correcting an error or updating an exception status appends a new `AuditEvent` entry with an incremented sequence number (`seq`), timestamp, and state metadata.
- **Strict Invariants**: `src/audit_log.py` enforces integer paise validation, schema versioning (`schema_version: 1`), and immutability (frozen dataclasses, no mutating methods exposed).

---

## 5. Developer Retrospective: The Pass 2 State Pipeline Bug

During early development, a subtle bug emerged in the interaction between Pass 2 and Pass 3.

### The Bug
Pass 2 was correctly identifying records that failed batch-sum matching (such as `variance_breach` and `ambiguous_amount` cases), but instead of returning them as terminal exception records, it was forwarding them downstream to Pass 3.

### Impact
When `bank_30` (an ambiguous amount matching 3 competing settlement batches) reached Pass 3, the AI parser attempted narration extraction on it. Depending on prompt execution timing, Pass 3 occasionally re-classified `bank_30` under a different exception label (`unparseable_narration_unresolvable`), creating non-deterministic classification behavior across runs.

### Root Cause & Fix
- **Root Cause**: Non-terminal control flow in Pass 2.
- **Fix**: Made Pass 2's failure classifications (`variance_breach`, `ambiguous_amount`) strictly terminal exception decisions, halting downstream pipeline execution for those records.

### Key Testing Lesson
In outcome-level grading (checking if `bank_30` became an exception), the system scored 100% because `bank_30` ended up in the exception queue either way. However, decision-level verification (checking *which* exception type was assigned and *why*) caught the bug. This reinforced the principle of auditing decision paths, not just binary pass/fail outcomes.

---

## 6. Trade-offs & Scope Management

### Scoping Out Trusted Execution Environments (TEE)
Hardware enclave execution (e.g., AWS Nitro Enclaves, Intel SGX) was evaluated for run-time state isolation. It was deliberately scoped out because:
1. An append-only, tamper-evident JSONL audit log with full sequence verification provides the necessary explainability and audit guarantees.
2. Hardware enclave attestation introduces heavy cloud infrastructure dependencies and deployment overhead that do not align with hackathon evaluation constraints.

### Scoping AI to Information Extraction
We intentionally avoided using LLMs for numerical reconciliation or financial calculations. LLMs are non-deterministic and susceptible to arithmetic hallucination. Keeping financial logic strictly in Python standard libraries while using Groq qwen/qwen3.6-27b solely for text extraction achieves speed, reliability, and zero arithmetic errors.

---

## 7. Production Readiness & Future Roadmap

To scale SettleSense from buildathon demonstration to enterprise production:

1. **Direct Webhook Ingestion**: Integrate live Razorpay Webhooks (`settlement.processed`, `payment.captured`, `refund.created`) to process settlements in real time rather than batch CSV polling.
2. **Core Banking API & SFTP Feeds**: Automate host-to-host (H2H) bank statement ingestion via SFTP / Open Banking APIs (e.g., ICICI CIB, HDFC Enets).
3. **Automated Triage Actions**: Allow finance teams to execute 1-click resolution actions from `dashboard.html` (e.g., auto-issuing gateway query tickets for variance breaches or approving tolerance write-offs).
4. **Distributed Task Execution**: Transition pipeline execution to distributed task queues (Celery / Redis) with a PostgreSQL backend for multi-tenant enterprise isolation.

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NormalizedRecord:
    record_id: str
    source: str
    date: datetime
    amount: float
    narration: str
    utr: Optional[str]
    payment_ids: List[str]
    settlement_id: Optional[str]
    fee: float
    gst_on_fee: float
    net_amount: float
    status: str
    raw_data: Dict


class BankStatementNormalizer:
    UTR_PATTERNS = [
        r'UTR([A-Z0-9]+)',
        r'UTR[_-]([A-Z0-9]+)',
        r'NEFT[_-]RAZORPAY[_-]UTR([A-Z0-9]+)',
        r'RTGS[_-]RAZORPAY[_-]UTR([A-Z0-9]+)',
        r'IMPS/(\d+)/RAZORPAY',
        r'UPI/CR/(\d+)/RAZORPAY',
    ]

    def __init__(self):
        self.records = []

    def parse_narration(self, narration: str) -> Dict:
        utr = None
        for pattern in self.UTR_PATTERNS:
            match = re.search(pattern, narration, re.IGNORECASE)
            if match:
                utr = match.group(1).upper()
                break

        amount_match = re.search(r'[₹]?\s*([\d,]+\.?\d*)', narration)
        amount = None
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))

        return {
            'utr': utr,
            'extracted_amount': amount,
            'clean_narration': narration.strip()
        }

    def normalize(self, df: pd.DataFrame) -> List[NormalizedRecord]:
        normalized = []
        for idx, row in df.iterrows():
            parsed = self.parse_narration(row['narration'])
            
            record = NormalizedRecord(
                record_id=f"bank_{idx}",
                source="bank",
                date=pd.to_datetime(row['date']),
                amount=float(row['amount']),
                narration=row['narration'],
                utr=parsed['utr'],
                payment_ids=[],
                settlement_id=None,
                fee=0.0,
                gst_on_fee=0.0,
                net_amount=float(row['amount']),
                status="pending",
                raw_data=row.to_dict()
            )
            normalized.append(record)
        return normalized


class RazorpaySettlementNormalizer:
    def normalize(self, df: pd.DataFrame) -> List[NormalizedRecord]:
        normalized = []
        for idx, row in df.iterrows():
            record = NormalizedRecord(
                record_id=f"razorpay_{idx}",
                source="razorpay",
                date=pd.to_datetime(row['settled_at']),
                amount=float(row['amount']),
                narration=f"Razorpay settlement {row['settlement_id']} payment {row['payment_id']}",
                utr=row['utr'] if row['utr'] and row['utr'] != '' else None,
                payment_ids=[row['payment_id']],
                settlement_id=row['settlement_id'],
                fee=float(row['fee']),
                gst_on_fee=float(row['gst_on_fee']),
                net_amount=float(row['net_amount']),
                status=row['status'],
                raw_data=row.to_dict()
            )
            normalized.append(record)
        return normalized


class OrderLedgerNormalizer:
    def normalize(self, df: pd.DataFrame) -> List[NormalizedRecord]:
        normalized = []
        for idx, row in df.iterrows():
            record = NormalizedRecord(
                record_id=f"order_{idx}",
                source="order",
                date=pd.to_datetime(row['created_at']),
                amount=float(row['amount']),
                narration=f"Order {row['order_id']} payment {row['payment_id']}",
                utr=None,
                payment_ids=[row['payment_id']],
                settlement_id=None,
                fee=0.0,
                gst_on_fee=0.0,
                net_amount=float(row['amount']),
                status=row['status'],
                raw_data=row.to_dict()
            )
            normalized.append(record)
        return normalized


class Normalizer:
    def __init__(self):
        self.bank_normalizer = BankStatementNormalizer()
        self.razorpay_normalizer = RazorpaySettlementNormalizer()
        self.order_normalizer = OrderLedgerNormalizer()

    def load_and_normalize(self, bank_path: str, razorpay_path: str, order_path: str) -> Dict[str, List[NormalizedRecord]]:
        bank_df = pd.read_csv(bank_path)
        razorpay_df = pd.read_csv(razorpay_path)
        order_df = pd.read_csv(order_path)

        return {
            'bank': self.bank_normalizer.normalize(bank_df),
            'razorpay': self.razorpay_normalizer.normalize(razorpay_df),
            'order': self.order_normalizer.normalize(order_df)
        }

    def to_dataframe(self, records: List[NormalizedRecord]) -> pd.DataFrame:
        return pd.DataFrame([{
            'record_id': r.record_id,
            'source': r.source,
            'date': r.date,
            'amount': r.amount,
            'narration': r.narration,
            'utr': r.utr,
            'payment_ids': '|'.join(r.payment_ids) if r.payment_ids else '',
            'settlement_id': r.settlement_id,
            'fee': r.fee,
            'gst_on_fee': r.gst_on_fee,
            'net_amount': r.net_amount,
            'status': r.status
        } for r in records])


if __name__ == "__main__":
    normalizer = Normalizer()
    data = normalizer.load_and_normalize(
        'data/bank_statement.csv',
        'data/razorpay_settlements.csv',
        'data/order_ledger.csv'
    )
    
    for source, records in data.items():
        df = normalizer.to_dataframe(records)
        print(f"\n{source.upper()} - {len(records)} records")
        print(df[['record_id', 'date', 'amount', 'utr', 'payment_ids', 'settlement_id', 'net_amount']].head(10))
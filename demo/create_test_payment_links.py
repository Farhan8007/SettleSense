#!/usr/bin/env python3
"""Create Razorpay test Payment Links for manual checkout seeding."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import razorpay


AMOUNTS_PAISE = [50000, 120000, 35000, 200000, 80000]


def format_rupees(amount_paise: int) -> str:
    return f"Rs {amount_paise / 100:.2f}"


def main() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    key_id = os.environ["RAZORPAY_KEY_ID"]
    key_secret = os.environ["RAZORPAY_KEY_SECRET"]

    client = razorpay.Client(auth=(key_id, key_secret))

    try:
        for index, amount_paise in enumerate(AMOUNTS_PAISE, start=1):
            link = client.payment_link.create({
                "amount": amount_paise,
                "currency": "INR",
                "accept_partial": False,
                "description": f"SettleSense test payment {index}",
                "customer": {
                    "name": f"SettleSense Test Customer {index}",
                    "email": f"settlesense.test{index}@example.com",
                    "contact": f"900000000{index}",
                },
                "notify": {
                    "sms": False,
                    "email": False,
                },
                "reminder_enable": False,
            })

            print(f"Payment link: {link['short_url']}")
            print(f"Amount: {format_rupees(amount_paise)} ({amount_paise} paise)")
            print(
                "Open this link, choose UPI, enter success@razorpay "
                "as the UPI ID to complete instantly"
            )
            print()
    except Exception as error:
        print(f"Razorpay Payment Link creation failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Generates a synthetic retail transactions dataset that mimics a real
raw data drop landing in a Bronze layer -- including realistic messiness
(nulls, duplicates, inconsistent types) so the Silver-layer cleaning and
data-quality checks have something real to catch.

Usage:
    python ingestion/generate_synthetic_data.py --rows 5000 --out data/raw/transactions_2026_08_09.csv
"""
import argparse
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

CATEGORIES = [
    "Electronics", "Home & Kitchen", "Fashion", "Groceries",
    "Beauty & Personal Care", "Sports & Outdoors", "Books", "Toys",
]

PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "cash_on_delivery", "wallet"]


def generate_customers(n_customers: int) -> pd.DataFrame:
    rows = []
    for i in range(n_customers):
        rows.append({
            "customer_id": f"CUST{i:06d}",
            "signup_date": fake.date_between(start_date="-3y", end_date="-30d"),
            "region": fake.state(),
            "age": random.choice([random.randint(18, 70), None]) if random.random() < 0.03 else random.randint(18, 70),
        })
    return pd.DataFrame(rows)


def generate_transactions(customers: pd.DataFrame, n_rows: int, dirty_fraction: float = 0.05) -> pd.DataFrame:
    rows = []
    customer_ids = customers["customer_id"].tolist()

    for _ in range(n_rows):
        cust = random.choice(customer_ids)
        category = random.choice(CATEGORIES)
        amount = round(random.uniform(5, 2000), 2)
        qty = random.randint(1, 5)
        ts = fake.date_time_between(start_date="-180d", end_date="now")

        row = {
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cust,
            "product_category": category,
            "quantity": qty,
            "unit_price": amount,
            "transaction_amount": round(amount * qty, 2),
            "payment_method": random.choice(PAYMENT_METHODS),
            "transaction_ts": ts.isoformat(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["quantity"] = df["quantity"].astype("object")  # allow mixed types for the injected "bad_type" case below

    # --- inject realistic dirtiness for the Silver layer / data quality checks to catch ---
    n_dirty = int(len(df) * dirty_fraction)
    dirty_idx = random.sample(range(len(df)), n_dirty)

    for idx in dirty_idx:
        issue = random.choice(["null_category", "negative_amount", "null_customer", "dup_row", "bad_type"])
        if issue == "null_category":
            df.at[idx, "product_category"] = None
        elif issue == "negative_amount":
            df.at[idx, "transaction_amount"] = -abs(df.at[idx, "transaction_amount"])
        elif issue == "null_customer":
            df.at[idx, "customer_id"] = None
        elif issue == "dup_row":
            df = pd.concat([df, df.loc[[idx]]], ignore_index=True)
        elif issue == "bad_type":
            df.at[idx, "quantity"] = "N/A"

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--customers", type=int, default=800)
    parser.add_argument("--out", type=str, default="data/raw/transactions.csv")
    parser.add_argument("--dirty-fraction", type=float, default=0.05)
    args = parser.parse_args()

    customers = generate_customers(args.customers)
    transactions = generate_transactions(customers, args.rows, args.dirty_fraction)

    customers.to_csv("data/raw/customers.csv", index=False)
    transactions.to_csv(args.out, index=False)

    print(f"Wrote {len(customers)} customers -> data/raw/customers.csv")
    print(f"Wrote {len(transactions)} transactions -> {args.out}")
    print(f"Injected ~{args.dirty_fraction:.0%} data quality issues for downstream Silver/DQ checks to catch")


if __name__ == "__main__":
    main()

"""One-off EDA probe: confirm the real schema before writing the pipeline."""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

txn_head = pd.read_csv(RAW / "ieee_train_transaction.csv", nrows=5)
print("=== train_transaction ===")
print("cols:", len(txn_head.columns))
print("first 60:", list(txn_head.columns[:60]))
print(txn_head[["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]])

ident_head = pd.read_csv(RAW / "ieee_train_identity.csv", nrows=5)
print("\n=== train_identity ===")
print("cols:", len(ident_head.columns))
print(list(ident_head.columns))

print("\n=== label rate + time span (streamed) ===")
cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card4", "card6"]
n = 0
fraud = 0
dt_min, dt_max = None, None
for chunk in pd.read_csv(RAW / "ieee_train_transaction.csv", usecols=cols, chunksize=200_000):
    n += len(chunk)
    fraud += int(chunk["isFraud"].sum())
    lo, hi = chunk["TransactionDT"].min(), chunk["TransactionDT"].max()
    dt_min = lo if dt_min is None else min(dt_min, lo)
    dt_max = hi if dt_max is None else max(dt_max, hi)

print(f"rows={n:,}  fraud={fraud:,}  rate={fraud / n:.4%}")
print(f"TransactionDT range: {dt_min:,} .. {dt_max:,}  ({(dt_max - dt_min) / 86400:.1f} days)")
print("\nProductCD / card4 / card6 value counts on last chunk:")
for c in ("ProductCD", "card4", "card6"):
    print(f"  {c}: {chunk[c].value_counts(dropna=False).head(8).to_dict()}")

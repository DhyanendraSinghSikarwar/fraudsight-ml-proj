"""Entity profiles the triage tools serve from.

IEEE-CIS is a flat transaction table: there is no customer table, no device
registry, and no merchant directory. Rather than pretend otherwise, this module
*derives* those records from the transaction history and names the approximation
in each profile's `basis` field, which is carried through to the brief. An
analyst reading "card1 is used as the customer key" can judge the evidence; an
analyst reading "customer profile" cannot.

Two constraints make the derived profiles honest:

* **Training window only.** Profiles are built from the rows before the embargo,
  the same rows the scorer trained on. Building them from all history would let
  a triage case for a test-window transaction cite facts from that
  transaction's own future -- which would inflate every eval number in Track B
  and be nearly invisible.

* **Historical outcomes are evidence, not features.** `prior_confirmed_fraud`
  comes from labels. It is deliberately *not* in `feature_names()` and never
  reaches the scorer; a fraud team genuinely knows which cards have charged
  back, and withholding that from the analyst would be modelling a world that
  does not exist. The separation is enforced by the tools reading this file and
  the scorer reading only `features/spec.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudsight.config import settings
from fraudsight.features.spec import AMOUNT, LABEL, TIME, _norm_key

SECONDS_PER_DAY = 86_400

# Entities with fewer than this many transactions get a profile with
# `thin_history: True`. Rates computed on 2 observations are noise, and an agent
# citing "100% historical fraud rate (1 of 1)" as strong evidence is exactly the
# failure the flag exists to prevent.
THIN_HISTORY_MIN = 5


@dataclass(frozen=True)
class ProfileTables:
    customers: dict[str, dict[str, Any]]
    devices: dict[str, dict[str, Any]]
    merchants: dict[str, dict[str, Any]]
    built_from: dict[str, Any]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("customers", "devices", "merchants"):
            (directory / f"{name}.json").write_text(
                json.dumps(getattr(self, name)), encoding="utf-8"
            )
        (directory / "profiles_meta.json").write_text(
            json.dumps(self.built_from, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | None = None) -> ProfileTables:
        d = directory or (settings.artifact_dir / "profiles")
        missing = [n for n in ("customers", "devices", "merchants") if not (d / f"{n}.json").exists()]
        if missing:
            raise FileNotFoundError(
                f"profile tables {missing} missing in {d} -- run `fraudsight profiles`"
            )
        return cls(
            customers=json.loads((d / "customers.json").read_text(encoding="utf-8")),
            devices=json.loads((d / "devices.json").read_text(encoding="utf-8")),
            merchants=json.loads((d / "merchants.json").read_text(encoding="utf-8")),
            built_from=json.loads((d / "profiles_meta.json").read_text(encoding="utf-8")),
        )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _day(seconds: float) -> float:
    return round(float(seconds) / SECONDS_PER_DAY, 2)


def build(train_parquet: Path | None = None) -> ProfileTables:
    path = train_parquet or (settings.data_dir / "processed" / "train.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make features` first")
    df = pd.read_parquet(path)

    df = df.assign(
        _card=df["card1"].map(_norm_key),
        _addr=df["addr1"].map(_norm_key),
        _device=df["DeviceInfo"].astype("object").map(_norm_key),
        _email=df["P_emaildomain"].astype("object").map(_norm_key),
        _product=df["ProductCD"].astype("object").map(_norm_key),
    )
    df["_merchant"] = df["_product"] + "|" + df["_addr"]

    customers = _customer_profiles(df)
    devices = _device_profiles(df)
    merchants = _merchant_profiles(df)

    meta = {
        "rows": int(len(df)),
        "window_days": [_day(df[TIME].min()), _day(df[TIME].max())],
        "source": str(path.name),
        "counts": {
            "customers": len(customers),
            "devices": len(devices),
            "merchants": len(merchants),
        },
        "thin_history_min": THIN_HISTORY_MIN,
        "note": (
            "Derived from the training split only. card1 stands in for a customer, "
            "DeviceInfo+id_31 for a device, ProductCD+addr1 for a merchant. IEEE-CIS "
            "ships none of these as first-class entities."
        ),
    }
    return ProfileTables(customers, devices, merchants, meta)


def _customer_profiles(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, g in df.groupby("_card", sort=False):
        n = len(g)
        amt = g[AMOUNT].astype("float64")
        fraud = int(g[LABEL].sum())
        out[str(key)] = {
            "customer_key": str(key),
            "basis": "card1 (issuer BIN-level card identifier)",
            "transactions": n,
            "first_seen_day": _day(g[TIME].min()),
            "last_seen_day": _day(g[TIME].max()),
            "tenure_days": _day(g[TIME].max() - g[TIME].min()),
            "amount_mean": round(float(amt.mean()), 2),
            "amount_median": round(float(amt.median()), 2),
            "amount_p95": round(float(amt.quantile(0.95)), 2),
            "amount_max": round(float(amt.max()), 2),
            "distinct_addresses": int(g["_addr"].nunique()),
            "distinct_devices": int(g["_device"].nunique()),
            "distinct_email_domains": int(g["_email"].nunique()),
            "top_product": str(g["_product"].mode().iloc[0]) if n else "unknown",
            "night_transaction_rate": _rate(int((g["is_night"] > 0).sum()), n),
            "prior_confirmed_fraud": fraud,
            "prior_fraud_rate": _rate(fraud, n),
            "thin_history": n < THIN_HISTORY_MIN,
        }
    return out


def _device_profiles(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, g in df.groupby("_device", sort=False):
        n = len(g)
        fraud = int(g[LABEL].sum())
        browsers = g["id_31"].astype("object").map(_norm_key)
        out[str(key)] = {
            "device_key": str(key),
            "basis": "DeviceInfo string, enriched with id_31 (browser/OS) and DeviceType",
            "transactions": n,
            "first_seen_day": _day(g[TIME].min()),
            "last_seen_day": _day(g[TIME].max()),
            "distinct_cards": int(g["_card"].nunique()),
            # Many cards on one device is the mule/cash-out signature. Reported
            # as a ratio because the absolute count is meaningless without volume.
            "cards_per_transaction": round(float(g["_card"].nunique()) / n, 4),
            "device_type": str(g["DeviceType"].astype("object").map(_norm_key).mode().iloc[0]),
            "top_browser": str(browsers.mode().iloc[0]) if n else "unknown",
            "distinct_browsers": int(browsers.nunique()),
            "amount_mean": round(float(g[AMOUNT].astype("float64").mean()), 2),
            "prior_confirmed_fraud": fraud,
            "prior_fraud_rate": _rate(fraud, n),
            "thin_history": n < THIN_HISTORY_MIN,
        }
    return out


def _merchant_profiles(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, g in df.groupby("_merchant", sort=False):
        n = len(g)
        amt = g[AMOUNT].astype("float64")
        fraud = int(g[LABEL].sum())
        out[str(key)] = {
            "merchant_key": str(key),
            "basis": "ProductCD x addr1 -- a merchant *proxy*; IEEE-CIS has no merchant column",
            "product_code": str(g["_product"].iloc[0]),
            "billing_region": str(g["_addr"].iloc[0]),
            "transactions": n,
            "amount_mean": round(float(amt.mean()), 2),
            "amount_p95": round(float(amt.quantile(0.95)), 2),
            "distinct_cards": int(g["_card"].nunique()),
            "prior_confirmed_fraud": fraud,
            "prior_fraud_rate": _rate(fraud, n),
            # The number an analyst actually wants: is this merchant's fraud rate
            # unusual for the portfolio, or is the whole book at 3.4%?
            "fraud_rate_vs_portfolio": round(
                _rate(fraud, n) / max(float(df[LABEL].mean()), 1e-9), 3
            ),
            "thin_history": n < THIN_HISTORY_MIN,
        }
    return out


def merchant_key(product: Any, addr: Any) -> str:
    return f"{_norm_key(product)}|{_norm_key(addr)}"


def portfolio_baseline(tables: ProfileTables) -> dict[str, float]:
    """Population statistics an agent needs to judge whether a number is odd.

    Without these, "this card has made 14 transactions" is uninterpretable and
    the model fills the gap with a prior it invented.
    """
    txns = np.array([c["transactions"] for c in tables.customers.values()], dtype=np.float64)
    amts = np.array([c["amount_mean"] for c in tables.customers.values()], dtype=np.float64)
    return {
        "median_transactions_per_customer": round(float(np.median(txns)), 2),
        "p90_transactions_per_customer": round(float(np.percentile(txns, 90)), 2),
        "median_customer_mean_amount": round(float(np.median(amts)), 2),
        "p95_customer_mean_amount": round(float(np.percentile(amts, 95)), 2),
        "portfolio_fraud_rate": round(
            float(
                sum(c["prior_confirmed_fraud"] for c in tables.customers.values())
                / max(sum(c["transactions"] for c in tables.customers.values()), 1)
            ),
            6,
        ),
    }

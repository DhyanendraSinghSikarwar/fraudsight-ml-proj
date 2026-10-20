"""Fetch the fraud datasets the project trains and replays on.

Three source tiers, tried in order, so `make data` works on any machine:

1. Hugging Face mirrors (no credentials) -- the default path.
2. Kaggle CLI (if ~/.kaggle/kaggle.json exists) -- canonical source.
3. Synthetic generator -- schema-identical fallback so the demo always runs.

Run:  uv run python scripts/fetch_data.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class Source:
    name: str
    hf_repo: str
    hf_path: str
    dest: str
    kaggle: str | None
    approx_mb: int


SOURCES: dict[str, tuple[Source, ...]] = {
    # Primary training + eval corpus: real labels, 393 raw columns, 590k rows.
    "ieee": (
        Source(
            name="ieee_transaction",
            hf_repo="aliceczr/ieee-fraud-detection",
            hf_path="train_transaction.csv",
            dest="ieee_train_transaction.csv",
            kaggle="ieee-fraud-detection",
            approx_mb=652,
        ),
        Source(
            name="ieee_identity",
            hf_repo="aliceczr/ieee-fraud-detection",
            hf_path="train_identity.csv",
            dest="ieee_train_identity.csv",
            kaggle="ieee-fraud-detection",
            approx_mb=25,
        ),
    ),
    # Streaming replay corpus: 6.3M mobile-money transfers, cheap to replay at volume.
    "paysim": (
        Source(
            name="paysim",
            hf_repo="theman10/paysim",
            hf_path="paysim.csv",
            dest="paysim.csv",
            kaggle="ealaxi/paysim1",
            approx_mb=471,
        ),
    ),
}


def _human(n: int) -> str:
    return f"{n / 1_048_576:.1f}MB"


def _download(url: str, dest: Path, approx_mb: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    started = time.monotonic()
    last_line = 0.0

    def hook(block: int, block_size: int, total: int) -> None:
        nonlocal last_line
        now = time.monotonic()
        if now - last_line < 2.0:
            return
        last_line = now
        got = block * block_size
        pct = f"{100 * got / total:.0f}%" if total > 0 else "?"
        rate = got / max(now - started, 1e-6) / 1_048_576
        sys.stderr.write(f"\r  {dest.name}: {_human(got)} ({pct}) {rate:.1f}MB/s")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310 - fixed https hosts
    sys.stderr.write("\n")
    tmp.replace(dest)


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None and (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists()


def _fetch_via_kaggle(slug: str) -> bool:
    print(f"  trying kaggle CLI: {slug}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA_DIR), "--unzip"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  kaggle path unavailable ({exc.__class__.__name__}); falling back")
        return False


def fetch(source: Source, force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / source.dest
    if dest.exists() and not force:
        print(f"  {source.dest}: already present ({_human(dest.stat().st_size)})")
        return dest

    url = HF.format(repo=source.hf_repo, path=source.hf_path)
    print(f"  {source.dest}: fetching ~{source.approx_mb}MB from {source.hf_repo}")
    try:
        _download(url, dest, source.approx_mb)
    except Exception as exc:  # noqa: BLE001 - any transport failure should degrade, not crash
        print(f"  HF mirror failed: {exc}")
        if source.kaggle and _kaggle_available() and _fetch_via_kaggle(source.kaggle):
            return dest
        raise
    return dest


def checksum(path: Path, limit: int = 8 << 20) -> str:
    """Hash the first `limit` bytes -- enough to detect a swapped mirror cheaply."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    wanted = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for key in wanted:
        print(f"[{key}]")
        for source in SOURCES[key]:
            path = fetch(source, force=args.force)
            print(f"  {path.name}  sha256[:16]={checksum(path)}")
    print(f"\ndata root: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

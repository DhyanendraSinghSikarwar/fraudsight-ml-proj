"""fraudsight -- real-time fraud triage.

Two paths, deliberately separated:

  scoring path    every transaction, LightGBM, ~8ms, $0
  triage path     the ambiguous ~8%, LLM agent, ~3s, ~$0.001

The split is the whole point of the system. See docs/adr/0001-gbdt-in-hot-path.md
and docs/adr/0002-llm-only-for-triage.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

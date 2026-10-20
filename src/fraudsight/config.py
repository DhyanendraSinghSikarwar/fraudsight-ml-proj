"""Central configuration. Secrets are SecretStr so a stray print or a trace
payload can never leak a key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["anthropic", "openai", "deterministic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- providers -------------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # "deterministic" is a real, shipped provider: a rule-based brief generator
    # used for CI, offline evals, and `make demo`. It is not a mock -- it
    # implements the same LLMClient protocol and is scored by the same judges,
    # which is what lets the eval gate run without a vendor account.
    primary_provider: Provider = "anthropic"
    fallback_provider: Provider | None = "deterministic"

    triage_model: str = "claude-opus-5"
    fallback_model: str = "gpt-4.1-mini"
    judge_model: str = "claude-opus-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    # ---- infrastructure --------------------------------------------------
    database_url: str = "postgresql://fraudsight:fraudsight@localhost:5432/fraudsight"
    redis_url: str = "redis://localhost:6379/0"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"

    # When postgres is not reachable we fall back to an on-disk store so the
    # whole pipeline stays runnable. See docs/adr/0003.
    vector_backend: Literal["pgvector", "local"] = "local"
    feature_backend: Literal["redis", "local"] = "local"

    # ---- routing bands ---------------------------------------------------
    # Defaults are placeholders; the real values are written by
    # `fraudsight thresholds` after minimising expected cost on the val split
    # and are loaded from artifacts/thresholds.json at serve time.
    auto_approve_below: float = Field(0.15, ge=0.0, le=1.0)
    auto_block_above: float = Field(0.95, ge=0.0, le=1.0)

    # Band-selection policy, not model hyperparameters. `band_min_recall` is the
    # NFR commitment; when the scorer cannot reach it at any review rate under
    # the cap, `select_operating_point` ships the cost-optimal bands and marks
    # the report `nfr_recall_met: False` rather than quietly lowering the floor.
    band_min_recall: float = Field(0.95, ge=0.0, le=1.0)
    band_max_review_rate: float = Field(0.15, ge=0.0, le=1.0)

    # ---- cost matrix (dollars) ------------------------------------------
    # Every one of these is a business input, not a model hyperparameter.
    # They are the reason recall is non-negotiable and precision is tradeable.
    fraud_loss_usd: float = 180.0  # average net loss on a missed fraud
    false_decline_usd: float = 25.0  # churn/goodwill cost of wrongly blocking
    analyst_review_usd: float = 4.20  # 6 min x $42/h loaded rate
    triage_llm_usd: float = 0.0012  # measured cost of one LLM triage

    # ---- budgets and limits ---------------------------------------------
    daily_llm_budget_usd: float = 20.0
    budget_alert_fraction: float = 0.8
    triage_timeout_s: float = 8.0
    triage_max_repairs: int = 2
    circuit_breaker_cooldown_s: float = 30.0
    max_tool_concurrency: int = 8

    # ---- paths -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    artifact_dir: Path = REPO_ROOT / "artifacts"
    trace_dir: Path = REPO_ROOT / "artifacts" / "traces"

    # ---- eval gate -------------------------------------------------------
    gate_min_agreement: float = 0.85
    gate_max_hallucination: float = 0.02
    gate_min_schema_pass: float = 0.99
    gate_min_judge_kappa: float = 0.60

    @field_validator("data_dir", "artifact_dir", "trace_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @model_validator(mode="after")
    def _bands_ordered(self) -> Settings:
        if self.auto_approve_below >= self.auto_block_above:
            raise ValueError(
                f"auto_approve_below ({self.auto_approve_below}) must be below "
                f"auto_block_above ({self.auto_block_above}); otherwise the triage band is empty"
            )
        return self

    def key_for(self, provider: Provider) -> SecretStr | None:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "deterministic": SecretStr("not-required"),
        }[provider]

    def available_providers(self) -> tuple[Provider, ...]:
        """Providers we actually hold credentials for. `deterministic` always is."""
        out: list[Provider] = []
        for p in ("anthropic", "openai"):
            if self.key_for(p) is not None:  # type: ignore[arg-type]
                out.append(p)  # type: ignore[arg-type]
        out.append("deterministic")
        return tuple(out)

    def require_provider(self) -> None:
        if self.primary_provider != "deterministic" and not self.key_for(self.primary_provider):
            raise RuntimeError(
                f"primary_provider={self.primary_provider} but no API key is set. "
                f"Set the matching key in .env, or set PRIMARY_PROVIDER=deterministic "
                f"to run offline."
            )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifact_dir, self.trace_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

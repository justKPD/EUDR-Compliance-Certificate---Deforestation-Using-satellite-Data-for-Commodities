"""
Centralised application configuration.

Priority order (highest first):
  1. Explicit environment variables set in the shell
  2. `.env` file in the current working directory
  3. `~/.eudr/.env` (user-level config, useful for GEE credentials)
  4. Built-in defaults

The module-level `settings` singleton is created lazily so that importing
from `eudr.config` never crashes if credentials have not been provided yet.
Call `settings` as a property or use `get_settings()` when you need the
validated object – it will raise a clear error only at that point.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_files() -> list[str]:
    """Return .env file candidates in priority order."""
    candidates = [".env", str(Path.home() / ".eudr" / ".env")]
    return [f for f in candidates if Path(f).exists()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files() or None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Earth Engine ────────────────────────────────────────────────────
    gee_project_id: str = Field(
        default="",
        description="GEE cloud project ID (required for satellite data acquisition)",
    )
    gee_service_account_key: str = Field(
        default="",
        description="Path to GEE service-account JSON key. Empty = user auth.",
    )

    # ── HuggingFace ───────────────────────────────────────────────────────────
    hf_token: str = Field(
        default="",
        description="HuggingFace access token for gated models (Gemma)",
    )

    # ── Model identifiers ─────────────────────────────────────────────────────
    prithvi_model_id: str = Field(
        default="ibm-nasa-geospatial/Prithvi-100M",
        description="HuggingFace model ID for the Prithvi geospatial backbone",
    )
    council_model_id: str = Field(
        default="google/gemma-2b-it",
        description="HuggingFace model ID for the Gemma compliance council LLM",
    )

    # ── Compliance thresholds (EUDR Article 3) ────────────────────────────────
    eudr_cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR reference date — deforestation after this date is non-compliant",
    )
    direct_loss_threshold: float = Field(
        default=5.0,
        description="Forest-loss % inside polygon that triggers RED (non-compliant)",
    )
    buffer_loss_threshold: float = Field(
        default=20.0,
        description="Forest-loss % in buffer zone that triggers YELLOW (medium risk)",
    )
    buffer_radius_km: float = Field(
        default=5.0,
        description="Buffer zone radius in kilometres",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── Output ────────────────────────────────────────────────────────────────
    report_output_dir: Path = Field(default=Path("./reports"))

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

    def require_gee(self) -> str:
        """Return the GEE project ID, raising a clear error if not configured."""
        pid = self.gee_project_id.strip()
        if not pid:
            raise RuntimeError(
                "GEE_PROJECT_ID is not set.\n"
                "  Option A: add GEE_PROJECT_ID=your-project to .env\n"
                "  Option B: export GEE_PROJECT_ID=your-project in your shell\n"
                "  Option C: run `python setup_auth.py` for guided setup"
            )
        return pid


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated Settings singleton (created on first call)."""
    return Settings()  # type: ignore[call-arg]


# Convenience alias — use this everywhere; never construct Settings() directly
settings = get_settings()

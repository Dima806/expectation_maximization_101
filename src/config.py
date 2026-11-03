"""Typed configuration loaded from config/settings.yaml (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class EMConfig(BaseModel):
    """EM loop hyperparameters."""

    max_iter: int = 200
    tol: float = 1e-6
    reg_covar: float = 1e-6
    n_init: int = 10


class DataConfig(BaseModel):
    """Synthetic-data generation settings."""

    seed: int = 42
    n_samples: int = 1500


class Settings(BaseSettings):
    """Top-level project settings."""

    model_config = SettingsConfigDict(env_prefix="EM101_", extra="ignore")

    seed: int = 42
    em: EMConfig = Field(default_factory=EMConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Load settings from YAML, falling back to defaults when absent."""
        path = path or CONFIG_PATH
        if path.exists():
            with path.open() as handle:
                raw = yaml.safe_load(handle) or {}
            return cls(**raw)
        return cls()

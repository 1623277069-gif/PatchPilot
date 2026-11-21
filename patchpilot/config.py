"""Operator configuration shared by the API and the worker."""

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

Mode = Literal["mock", "base", "lora"]
Profile = Literal["demo", "unittest", "pytest"]
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(r"https://github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9_.-]+?)(?:\.git)?")


def repo_url(value: str) -> str:
    match = REPO_RE.fullmatch(value)
    if not match or match[2] in {".", ".."} or len(match[2]) > 100:
        raise ValueError("Use an HTTPS GitHub owner/repository URL")
    return f"https://github.com/{match[1].lower()}/{match[2].lower()}.git"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PATCHPILOT_", env_file=".env", extra="ignore")

    database_url: SecretStr
    api_token: SecretStr = Field(min_length=24)
    state_dir: Path = Path("./data/platform")
    model_url: str = "http://127.0.0.1:8000/v1"
    base_model: str = "google/gemma-4-31B-it"
    lora_model: str = ""
    model_token: SecretStr | None = None
    github_token: SecretStr | None = None
    embedding_model: str = ""
    embedding_cache: Path = Path("./data/embeddings")
    sandbox_image: str = "patchpilot-sandbox:local"
    sandbox_runtime: str = ""
    model_timeout: int = Field(default=75, ge=1, le=300)
    test_timeout: int = Field(default=45, ge=1, le=300)
    max_seconds: int = Field(default=420, ge=1, le=3600)

    @model_validator(mode="after")
    def validate_database(self):
        url = make_url(self.database_url.get_secret_value())
        if url.drivername != "mysql+pymysql" or not url.database:
            raise ValueError("Use mysql+pymysql with an application database name")
        return self

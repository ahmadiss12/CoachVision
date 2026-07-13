"""Runtime configuration for backend service."""

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_JWT_SECRETS = {"", "change-me"}


class Settings(BaseSettings):
    app_name: str = "CoachVision Backend"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/v1"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/coachvision"
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 30
    refresh_token_exp_days: int = 7

    @model_validator(mode="after")
    def require_secure_jwt_secret(self) -> "Settings":
        if self.environment != "development" and self.jwt_secret_key in INSECURE_JWT_SECRETS:
            raise ValueError(
                "JWT_SECRET_KEY must be set to a strong random value when "
                f"ENVIRONMENT is {self.environment!r}. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return self

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


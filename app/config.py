from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "STRATA DSS API"
    debug: bool = False
    dataset_path: str = "data/dataset (1).csv"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

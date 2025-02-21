from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    logfire_token: str | None = None
    logfire_send_to_logfire: bool = True
    groq_api_key: str | None = None

    class Config:
        # Looks for .env file in the project root directory
        # You can also use absolute path if needed:
        # env_file = "../.env"  # relative to this file
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

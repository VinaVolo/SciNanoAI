from src.utils.paths import get_project_path
from pydantic import AnyHttpUrl, Field, DirectoryPath
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    """
    Create static settings for project
    """

    PROJECT_DIR_PATH: DirectoryPath = Field(get_project_path(), env="PROJECT_DIR_PATH")

    S3_URL: AnyHttpUrl = Field("https://storage.yandexcloud.net", env="S3_URL")
    S3_BUCKET: str = Field("paper-storage", env="S3_BUCKET")
    S3_ACCESS_KEY: str = Field(..., env="S3_ACCESS_KEY")
    S3_SECRET_KEY: str = Field(..., env="S3_SECRET_KEY")


load_dotenv()
settings = Settings()
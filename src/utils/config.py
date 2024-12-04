from src.utils.paths import get_project_path
from pydantic import AnyHttpUrl, Field, DirectoryPath
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    """
    Create static settings for project
    """

    PROJECT_DIR_PATH: DirectoryPath = Field(get_project_path(), env="PROJECT_DIR_PATH")
    
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_API_BASE: str = Field(..., env="OPENAI_API_BASE")
    # S3_ACCESS_KEY: str = Field(..., env="S3_ACCESS_KEY")
    # S3_SECRET_KEY: str = Field(..., env="S3_SECRET_KEY")
    # S3_URL: str = Field(..., env="S3_URL")
    # FAISS_BUCKET: str = Field(..., env="FAISS_BUCKET")


load_dotenv()
settings = Settings()
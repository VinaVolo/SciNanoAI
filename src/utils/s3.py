import boto3
from botocore.config import Config
from src.utils.config import settings


def create_s3_session():
    """
    Creates s3 session
    :return: s3 session
    """
    session = boto3.Session(
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    s3 = session.client(
        "s3", endpoint_url=settings.S3_URL, config=Config(signature_version="s3v4")
    )
    return s3


def download_file_from_s3(s3_uri: str, path_to_local_storage: str):
    """
    Download file from s3 bucket

    :param s3_uri: Path to file in s3 bucket.
    :param path_to_local_storage: Path to file in local storage
    """
    s3 = create_s3_session()
    s3.download_file(settings.S3_BUCKET, s3_uri, path_to_local_storage)
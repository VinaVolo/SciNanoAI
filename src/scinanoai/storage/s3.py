"""Idempotent S3 download/upload helper for Yandex Cloud Object Storage."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_LOG = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://storage.yandexcloud.net"


@dataclass(frozen=True)
class S3Config:
    bucket: str
    access_key: str
    secret_key: str
    endpoint_url: str = _DEFAULT_ENDPOINT


def _config_from_env(bucket: str | None = None) -> S3Config:
    bucket_name = bucket or os.environ.get("S3_BUCKET")
    if not bucket_name:
        raise RuntimeError("S3_BUCKET environment variable is not set")
    access = os.environ.get("S3_ACCESS_KEY")
    secret = os.environ.get("S3_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("S3_ACCESS_KEY and S3_SECRET_KEY must be set")
    return S3Config(bucket=bucket_name, access_key=access, secret_key=secret)


class S3Client:
    """Thin wrapper around boto3 with sane retries and idempotent transfers."""

    def __init__(self, config: S3Config | None = None) -> None:
        from botocore.client import Config as BotoConfig
        from botocore.exceptions import ClientError  # re-export for callers

        self._config = config or _config_from_env()
        self._ClientError = ClientError

        import boto3

        session = boto3.session.Session(
            aws_access_key_id=self._config.access_key,
            aws_secret_access_key=self._config.secret_key,
        )
        boto_cfg = BotoConfig(
            signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}
        )
        self._resource = session.resource(
            "s3", endpoint_url=self._config.endpoint_url, config=boto_cfg
        )
        self._client = session.client("s3", endpoint_url=self._config.endpoint_url, config=boto_cfg)
        self._bucket = self._resource.Bucket(self._config.bucket)

    # -- read --------------------------------------------------------------
    def _iter_objects(self, prefix: str) -> Iterator:
        yield from self._bucket.objects.filter(Prefix=prefix)

    def _remote_size(self, key: str) -> int | None:
        try:
            obj = self._client.head_object(Bucket=self._config.bucket, Key=key)
            return int(obj["ContentLength"])
        except self._ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

    # -- public ops --------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8.0),
        retry=retry_if_exception_type(Exception),
    )
    def download_prefix(self, prefix: str, dest_root: Path) -> int:
        dest_root = Path(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for obj in self._iter_objects(prefix):
            if obj.key.endswith("/"):
                continue
            local_path = dest_root / obj.key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists() and local_path.stat().st_size == obj.size:
                _LOG.debug("Skipping %s — already present.", obj.key)
                continue
            _LOG.info("Downloading %s -> %s", obj.key, local_path)
            self._bucket.download_file(obj.key, str(local_path))
            downloaded += 1
        _LOG.info("Downloaded %d files under prefix=%s", downloaded, prefix)
        return downloaded

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8.0),
        retry=retry_if_exception_type(Exception),
    )
    def upload_folder(self, folder: Path, prefix: str = "") -> int:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(folder)
        uploaded = 0
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            key = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
            local_size = path.stat().st_size
            if self._remote_size(key) == local_size:
                continue
            _LOG.info("Uploading %s -> s3://%s/%s", path, self._config.bucket, key)
            self._bucket.upload_file(str(path), key)
            uploaded += 1
        _LOG.info("Uploaded %d files under prefix=%s", uploaded, prefix)
        return uploaded

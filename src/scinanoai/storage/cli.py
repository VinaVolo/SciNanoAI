"""Unified CLI for S3 download/upload (replaces three legacy scripts)."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from ..utils.logging import setup_logging
from ..utils.paths import get_project_root
from .s3 import S3Client, S3Config

_LOG = setup_logging("scinanoai.s3")


def _resolve_dest(value: str | None) -> Path:
    if value is None:
        return get_project_root()
    path = Path(value)
    return path if path.is_absolute() else get_project_root() / path


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="SciNanoAI S3 helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Idempotent download of a prefix")
    p_dl.add_argument("--prefix", required=True)
    p_dl.add_argument("--dest", default=None, help="Local destination (defaults to project root)")
    p_dl.add_argument("--bucket", default=None)

    p_up = sub.add_parser("upload", help="Idempotent upload of a folder")
    p_up.add_argument("--folder", required=True, type=Path)
    p_up.add_argument("--prefix", default="")
    p_up.add_argument("--bucket", default=None)

    args = parser.parse_args(argv)

    config = S3Config(
        bucket=args.bucket or "",
        access_key="__unused__",
        secret_key="__unused__",
    ) if args.bucket else None

    if args.command == "download":
        client = S3Client() if config is None else S3Client(config)
        client.download_prefix(args.prefix, _resolve_dest(args.dest))
    elif args.command == "upload":
        client = S3Client() if config is None else S3Client(config)
        folder = args.folder if args.folder.is_absolute() else get_project_root() / args.folder
        client.upload_folder(folder, args.prefix)


if __name__ == "__main__":
    main()

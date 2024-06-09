import os

from typing import Optional

import boto3
from botocore.client import Config


def get_minio_client():
    # MinIO server credentials
    minio_url = os.environ.get("MINIO_URL")
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")

    # Create an S3 client with MinIO configuration
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )
    return s3_client


def save_file_to_s3(
    local_path,
    bucket_name: Optional[str] = None,
    s3_folder_path: str = "./",
):
    """ "
    Save a file to MINIO bucket.

    Args:
        local_path (str): Local file path to save
        bucket_name (str): Bucket name to save
        save_dir (str): Local directory to save
    """
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert (
        os.environ.get("MINIO_BUCKET") is not None
    ), "MINIO_BUCKET environment variable must be set"

    # get the filename with extension from the local path
    filename = os.path.basename(local_path)

    s3_key = os.path.join(s3_folder_path, filename)

    s3_client = get_minio_client()

    s3_client.upload_file(local_path, bucket_name, s3_key)

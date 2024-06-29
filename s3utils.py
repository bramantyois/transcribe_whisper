import os

from typing import Optional

import argparse

import boto3
from botocore.client import Config

from dotenv import load_dotenv

load_dotenv()


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
    s3_folder_path: Optional[str] = None,
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

    if s3_folder_path is None:
        s3_key = local_path
    else:
        s3_key = os.path.join(s3_folder_path, filename)

    s3_client = get_minio_client()

    s3_client.upload_file(local_path, bucket_name, s3_key)


def save_folder_to_s3(
    local_folder_path,
    bucket_name=None,
    s3_folder_path: Optional[str] = None,
):
    """
    Save a folder to MINIO bucket.

    Args:
        local_folder_path (str): Local folder path to save
        bucket_name (str): Bucket name to save
        s3_folder_path (str): S3 folder path to save
    """
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert (
        os.environ.get("MINIO_BUCKET") is not None
    ), "MINIO_BUCKET environment variable must be set"

    s3_client = get_minio_client()

    if s3_folder_path is None:
        s3_folder_path = local_folder_path

    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_key = os.path.join(s3_folder_path, relative_path)
            s3_client.upload_file(local_path, bucket_name, s3_key)


def load_file_from_s3(s3_file_path, bucket_name=None, save_dir="./"):
    """
    Load a file from MINIO bucket.

    Args:
        s3_file_path (str): S3 file path to load
        bucket_name (str): Bucket name to load
        save_dir (str): Local directory to save
    """
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert (
        os.environ.get("MINIO_BUCKET") is not None
    ), "MINIO_BUCKET environment variable must be set"

    s3_client = get_minio_client()

    # creating dir
    file_dir = os.path.dirname(s3_file_path)
    save_file_dir = os.path.join(save_dir, file_dir)
    
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    s3_client.download_file(
        bucket_name, s3_file_path, os.path.join(save_dir, s3_file_path)
    )


def load_folder_from_s3(s3_folder_path, bucket_name=None, save_dir="./"):
    """
    Load a folder from MINIO bucket.

    Args:
        s3_folder_path (str): S3 folder path to load
        bucket_name (str): Bucket name to load
        save_dir (str): Local directory to save
    """
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert (
        os.environ.get("MINIO_BUCKET") is not None
    ), "MINIO_BUCKET environment variable must be set"
    local_folder_path = os.path.join(save_dir, s3_folder_path)
    s3_client = get_minio_client()

    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_path)
    for obj in objects.get("Contents", []):
        local_file_path = os.path.join(
            local_folder_path, os.path.relpath(obj["Key"], s3_folder_path)
        )
        if not os.path.exists(local_file_path):
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            s3_client.download_file(bucket_name, obj["Key"], local_file_path)


def get_list_of_files_s3(s3_folder_path, return_size=False, bucket_name=None):
    """
    Get list of files on S3 bucket

    Args:
        s3_folder_path (str): S3 folder path to load
        bucket_name (str): Bucket name to load
    """
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert (
        os.environ.get("MINIO_BUCKET") is not None
    ), "MINIO_BUCKET environment variable must be set"

    s3_client = get_minio_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path)

    files = []
    sizes = []

    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj["Key"])
            sizes.append(obj["Size"])
        
    if return_size:
        return files, sizes

    return files


class S3Mixin:
    def save(self, directory_path: str):
        """
        Save the model to a directory path. this method should save both the model and metadata
        """
        NotImplementedError

    def load(self, directory_path: str):
        """
        Load the model from a directory path. this method should load both the model and metadata
        """
        NotImplementedError

    def save_s3(
        self,
        bucket_name=None,
        local_folder_path: str = "models/nn_model/",
        s3_folder_path: Optional[str] = None,
    ):
        """
        Save the model to a MINIO bucket. This method should save both the model and metadata

        Args:
            bucket_name (str): Bucket name to save
            local_folder_path (str): Local directory to save
            s3_folder_path (str): S3 folder path to save. Default is the same as local_folder_path
        """
        if bucket_name is None:
            bucket_name = os.environ.get("MINIO_BUCKET")
        assert (
            os.environ.get("MINIO_BUCKET") is not None
        ), "MINIO_BUCKET environment variable must be set"

        if s3_folder_path is None:
            s3_folder_path = local_folder_path

        self.save(local_folder_path)

        s3_client = get_minio_client()

        for root, dirs, files in os.walk(local_folder_path):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder_path)
                s3_key = os.path.join(s3_folder_path, relative_path)
                s3_client.upload_file(local_path, bucket_name, s3_key)

    @classmethod
    def load_s3(
        cls,
        bucket_name=None,
        local_folder_path: str = "models/nn_model/",
        s3_folder_path: str = None,
    ):
        """
        Load the model from a MINIO bucket. This method should load both the model and metadata

        Args:
            bucket_name (str): Bucket name to load
            local_folder_path (str): Local directory to save
            s3_folder_path (str): S3 folder path to load. Default is the same as local_folder_path

        Returns:
            cls: Model class
        """
        if bucket_name is None:
            bucket_name = os.environ.get("MINIO_BUCKET")

        if s3_folder_path is None:
            s3_folder_path = local_folder_path

        assert (
            os.environ.get("MINIO_BUCKET") is not None
        ), "MINIO_BUCKET environment variable must be set"
        s3_client = get_minio_client()

        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_path)
        for obj in objects.get("Contents", []):
            local_file_path = os.path.join(
                local_folder_path, os.path.relpath(obj["Key"], s3_folder_path)
            )
            if not os.path.exists(local_file_path):
                if not os.path.exists(os.path.dirname(local_file_path)):
                    os.makedirs(os.path.dirname(local_file_path))
                s3_client.download_file(bucket_name, obj["Key"], local_file_path)

        return cls.load(local_folder_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["load_folder", "save_folder", "load_file", "save_file"],
        required=True,
    )

    parser.add_argument(
        "--path",
        type=str,
        help="can be S3 path or local path to load or save",
        required=True,
    )

    parser.add_argument("--bucket_name", type=str, help="bucket name", required=False)

    # if args.load_folder:
    #     load_folder_from_s3(args.s3_path, save_dir=args.local_path)
    # elif args.save_folder:
    #     save_folder_to_s3(args.local_path, s3_folder_path=args.s3_path)

    parser = parser.parse_args()

    if parser.mode == "load_folder":
        load_folder_from_s3(parser.path)
    elif parser.mode == "save_folder":
        save_folder_to_s3(parser.path)
    elif parser.mode == "load_file":
        load_file_from_s3(parser.path)
    elif parser.mode == "save_file":
        save_file_to_s3(parser.path)
    else:
        raise ValueError("Invalid mode")

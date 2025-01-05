import os
from tqdm import tqdm
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

def upload_folder_to_s3(folder_path, bucket_name, s3_prefix=""):
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY")
    )

    s3 = session.resource(
        service_name='s3',
        endpoint_url="https://storage.yandexcloud.net",
        config=Config(signature_version='s3v4')
    )

    s3_client = session.client(
        service_name='s3',
        endpoint_url="https://storage.yandexcloud.net",
        config=Config(signature_version='s3v4')
    )

    bucket = s3.Bucket(bucket_name)

    for root, dirs, files in tqdm(os.walk(folder_path)):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    bucket.upload_file(local_file_path, s3_key)
                else:
                    print(f"Error checking {s3_key}: {e}")

# Параметры
folder_path = "./data/Статьи для базы данных"
bucket_name = "scinanoai-faiss-db"
s3_prefix = "data/Статьи для базы данных"

# Вызов функции
upload_folder_to_s3(folder_path, bucket_name, s3_prefix)

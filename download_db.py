import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

def download_all_files_from_s3_bucket():
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY")
    )
    
    s3 = session.resource(
        service_name='s3',
        endpoint_url="https://storage.yandexcloud.net",
        config=Config(signature_version='s3v4')
    )
    
    bucket = s3.Bucket("scinanoai-faiss-db")
    
    for obj in bucket.objects.filter(Prefix="db/"):
        if obj.key.endswith('/'):
            print(f"Skipping directory placeholder: {obj.key}")
            continue

        local_path = os.path.join(".", obj.key)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"Downloading: {obj.key} -> {local_path}")
        bucket.download_file(obj.key, local_path)

download_all_files_from_s3_bucket()

import torch
import boto3
from io import BytesIO

def load_checkpoint_from_s3(bucket_name: str, s3_file_key: str):
    s3 = boto3.client('s3')
    with BytesIO() as f:
        s3.download_fileobj(bucket_name, s3_file_key, f)
        f.seek(0)
        checkpoint = torch.load(f)
    return checkpoint


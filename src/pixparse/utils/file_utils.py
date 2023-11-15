import torch
import fsspec
from typing import Optional

def clean_state_dict(state_dict):
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

def load_checkpoint(path: str, map_location='cpu'):
    with fsspec.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)
    return checkpoint


def check_exists(path: str):
    try:
        with fsspec.open(path):
            pass
    except FileNotFoundError:
        return False
    return True

def get_latest_checkpoint(path: str, pattern: str = "*.pt") -> Optional[str]:
    """
    Gets the latest checkpoint file from a given dir (local or s3)
    Args:
        directory (str): The directory to search for checkpoint files.
        pattern (str): Pattern to match checkpoint files, default is '*.pt'.

    Returns:
        The path to the latest checkpoint file, or None if no files are found.
    """
    fs = fsspec.filesystem('file') if path.startswith('/') else fsspec.filesystem('s3') 

    files = fs.glob(f'{path}/{pattern}')
    files_with_mtime = [(file, fs.info(file)['mtime']) for file in files]
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    return files_with_mtime[0][0] if files_with_mtime else None

import torch
import fsspec


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

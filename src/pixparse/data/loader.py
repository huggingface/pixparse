from chug import create_wds_loader, create_doc_anno_pipe, create_image_text_pipe
from chug.common import LoaderBundle, SharedCount

from .config import DatasetCfg

from typing import Callable

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from datasets import VerificationMode
from pixparse.data.datasets_utils import SafeDataset

from datasets import load_dataset

class GenericLoader(DataLoader):
    """
    supercharged dataloader for hf datasets to match methods from webdataset loaders. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size != 0:
            self.num_batches += 1
        
def create_loader(
    cfg: DatasetCfg,
    is_train: bool,
    image_preprocess,
    anno_preprocess,
    collate_fn: Callable = None,
    image_key="pdf;tif;tiff;png;jpg;jpeg",  # FIXME jpeg added for test w/ cc12m
    image_fmt="L",
    start_interval: int = 0,
    seed: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    create_decoder_pipe: Callable = create_doc_anno_pipe,
):
    """
    Creates a dataloader for training or validation based on configuration settings.

    Parameters:
        cfg (DatasetCfg): Configuration object for the dataset.
        is_train (bool): Indicates if the loader is for training data (True) or validation data (False).
        collate_fn (Callable): Collate function to be used in loader. 
        image_preprocess (Callable): Image preprocessing sequence.
        anno_preprocess (Callable): Annotation preprocessing sequence.
        image_key (str, optional): Image formats/extensions that can be recognized and processed.
            Default includes formats such as "pdf", "tif", "tiff", etc.
        image_fmt (str, optional): Image format for reading images. Default is "L" (8-bit pixels, black and white).
        seed (int, optional): Seed for random operations to ensure reproducibility. Default is 0.
        world_size (int, optional): Total number of processes in the distributed setup. Default is 1.
        local_rank (int, optional): Rank of the current process in the distributed setup. Default is 0.
        create_decoder_pipe (Callable, optional): Function to create the annotation decoder pipeline for json documents.
            Default is `create_doc_anno_pipe`.

    Returns:
        DataLoader: A PyTorch DataLoader instance configured according to the provided settings.

    Note:
        Currently supports "webdataset" and "hf_dataset" as dataset formats.
    """
    decoder = create_decoder_pipe(
        image_preprocess=image_preprocess,
        anno_preprocess=anno_preprocess,
        image_key=image_key,
        image_fmt=image_fmt,
    )
    # TODO afdd factory for dataloaders?
    if cfg.format == "webdataset":
        loader = create_wds_loader(
            cfg.source,
            decoder,
            is_train=is_train,
            num_samples=cfg.num_samples,
            workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            seed=seed,
            world_size=world_size,
        )
    elif cfg.format == "hf_dataset":
        # In the case of hf datasets, we use the collator defined at task level
        dataset = load_dataset(cfg.source, verification_mode=VerificationMode.ALL_CHECKS)[cfg.split]
        dataset = SafeDataset(dataset)
        training_sampler = DistributedSampler(
            dataset, rank=local_rank, shuffle=True, seed=seed, num_replicas=world_size, drop_last=True
        ) # FIXME should be global_rank
        if is_train:
            # create a shared epoch store to sync epoch to dataloader worker proc
            shared_interval_count = SharedCount(count=start_interval)
        else:
            shared_interval_count = None
        num_batches = len(dataset) // cfg.batch_size

        base_loader = DataLoader(
            dataset=dataset, 
            collate_fn=collate_fn,
            sampler=training_sampler, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers,
            )
        loader = LoaderBundle(
        loader=base_loader,
        num_batches=num_batches,
        num_samples=cfg.num_samples,
        shared_interval=shared_interval_count,
    )
    return loader

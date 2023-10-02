from typing import Callable

from chug import create_wds_loader, create_doc_anno_pipe
from chug.common import LoaderBundle
from torch.utils.data import DataLoader, DistributedSampler

from pixparse.data.datasets_utils import SafeDataset, CustomVQADataset
from .config import DataCfg


class BaseCollate:
    def __init__(self, tokenizer, image_preprocess, start_token: str, max_length:int=512):
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self.start_token = start_token
        self.max_length = max_length

    def tokenizer_fn(self, x):
        return self.tokenizer(
            x,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids[0]

    def __call__(self, batch):
        # TODO add item["image"], item["label"] as default?
        raise NotImplementedError("This method should be overridden by child classes")


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
    cfg: DataCfg,
    is_train: bool,
    image_preprocess,
    anno_preprocess,
    collate_fn: Callable = None,
    image_key="pdf;tif;tiff;png;jpg;jpeg",  # FIXME jpeg added for test w/ cc12m
    image_fmt="L",
    start_interval: int = 0,
    seed: int = 0,
    world_size: int = 1,
    global_rank: int = 0,
    create_decoder_pipe: Callable = create_doc_anno_pipe,
) -> LoaderBundle:
    """
    Creates a dataloader for training or validation based on configuration settings.

    Parameters:
        cfg (DataCfg): Configuration object for the dataset.
        is_train (bool): Indicates if the loader is for training data (True) or validation data (False).
        collate_fn (Callable): Collate function to be used in loader.
        image_preprocess (Callable): Image preprocessing sequence.
        anno_preprocess (Callable): Annotation preprocessing sequence.
        image_key (str, optional): Image formats/extensions that can be recognized and processed.
            Default includes formats such as "pdf", "tif", "tiff", etc.
        image_fmt (str, optional): Image format for reading images. Default is "L" (8-bit pixels, black and white).
        seed (int, optional): Seed for random operations to ensure reproducibility. Default is 0.
        world_size (int, optional): Total number of processes in the distributed setup. Default is 1.
        global_rank (int, optional): Rank of the current process in the distributed setup. Default is 0.
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
        from datasets import VerificationMode
        from datasets import load_dataset

        # In the case of hf datasets, we use the collator defined at task level
        if cfg.source == "SinglePageDocVQA":
            dataset = CustomVQADataset(
                root_dir=f"/fsx/pablo/.cache/{cfg.source}",  # FIXME hacky hack
                split=cfg.split,
            )
        else:
            dataset = load_dataset(
                cfg.source, verification_mode=VerificationMode.ALL_CHECKS
            )[cfg.split]
        dataset = SafeDataset(dataset)

        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                rank=global_rank,
                shuffle=True,
                seed=seed,
                num_replicas=world_size,
                drop_last=True,
            )

        base_loader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        loader = LoaderBundle(
            loader=base_loader,
            num_batches=len(base_loader),
            num_samples=len(dataset),
            sampler=sampler,
        )
    return loader


from chug import create_wds_loader, create_doc_anno_pipe

from .config import DatasetCfg


def create_loader(
        cfg: DatasetCfg,
        is_train: bool,
        image_preprocess,
        anno_preprocess,
        image_key='pdf;tif;tiff;png;jpg;jpeg',  # FIXME jpeg added for test w/ cc12m
        image_fmt='L',
        seed: int = 0,
        world_size: int = 1,
):
    decoder = create_doc_anno_pipe(
        image_preprocess=image_preprocess,
        anno_preprocess=anno_preprocess,
        image_key=image_key,
        image_fmt=image_fmt,
    )

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
    return loader

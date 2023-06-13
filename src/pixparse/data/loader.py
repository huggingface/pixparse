
from chug import create_wds_loader, create_doc_anno_pipe

from .config import DatasetCfg


def create_loader(
        cfg: DatasetCfg,
        is_train: bool,
        image_preprocess,
        anno_preprocess=None,
):
    decoder = create_doc_anno_pipe(
        image_preprocess=image_preprocess,
        anno_preprocess=anno_preprocess,
        image_key='tif;tiff;png;jpg;jpeg',  # FIXME jpeg added for test w/ cc12m
        image_fmt='RGB',  # FIXME RGB for test w/ cc12m
    )

    loader = create_wds_loader(
        cfg.source,
        decoder,
        is_train=is_train,
        num_samples=cfg.num_samples,
        batch_size=cfg.batch_size,
    )
    return loader

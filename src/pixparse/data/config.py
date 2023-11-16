from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


def image_fmt_to_chs(fmt: str):
    assert fmt in ('L', 'RGB')   # could support more...
    return 1 if fmt == 'L' else 3


@dataclass
class ImageInputCfg:
    image_size: Optional[Tuple[int, int]]
    image_mean: Union[float, Tuple[float, ...]] = None
    image_std: Union[float, Tuple[float, ...]] = None
    image_fmt: str = 'L'

    @property
    def image_chs(self):
        return image_fmt_to_chs(self.image_fmt)

    def __post_init__(self):
        if self.image_mean is None:
            self.image_mean = (0.5,) * self.image_chs
        elif self.image_chs == 1 and len(self.image_mean) > self.image_chs:
            self.image_mean = sum(self.image_mean) / len(self.image_mean),
        if self.image_std is None:
            self.image_std = (0.5,) * self.image_chs
        elif self.image_chs == 1 and len(self.image_std) > self.image_chs:
            self.image_std = sum(self.image_std) / len(self.image_std)


# text and vision preproc cfg separate? flattened into data cfg or move to model/task?
@dataclass
class PreprocessCfg:
    image_input: ImageInputCfg = field(default_factory=ImageInputCfg)
    aug_type: str = 'basic'


@dataclass
class DataCfg:
    source: str
    num_samples: int
    batch_size: int
    split: str = ""
    format: str = "webdataset"  # e.g. "hf_dataset" or "webdataset"
    num_workers: int = 4
    resampled: bool = False  # sample shards with replacement

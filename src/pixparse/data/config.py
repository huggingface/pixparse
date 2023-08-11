from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessCfg:  # text and vision preproc cfg separate? flattened into data cfg or move to model/task?
    # FIXME preprocessing hard coded in Task right now
    pass


@dataclass
class DatasetCfg:
    source: str
    format: str
    num_samples: int
    batch_size: int
    num_workers: int = 4


@dataclass
class DataCfg:
    train: Optional[DatasetCfg] = None
    eval: Optional[DatasetCfg] = None


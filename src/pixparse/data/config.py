from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessCfg:  # text and vision preproc cfg separate? flattened into data cfg or move to model/task?
    # FIXME preprocessing hard coded in Task right now
    pass


@dataclass
class DatasetCfg:
    source: str
    num_samples: int
    batch_size: int


@dataclass
class DataCfg:
    train: DatasetCfg
    eval: Optional[DatasetCfg] = None


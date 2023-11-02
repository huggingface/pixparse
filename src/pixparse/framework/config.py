from dataclasses import dataclass, field
from typing import Optional, Tuple

from pixparse.models import ModelArgs
from pixparse.tokenizers import TokenizerCfg


@dataclass
class OptimizationCfg:
    """
    This dataclass serves among others the task.opt arg set.
    """
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    learning_rate: float = 1e-4
    warmup_learning_rate: float = 0.
    weight_decay: float = .02
    eps: float = 1e-6
    clip_grad_value: Optional[float] = None  # Gradient clipping value.
    clip_grad_mode: Optional[str] = None  # Gradient clipping mode, one of ('norm', 'value', 'agc').
    grad_accum_steps: int = 1
    grad_checkpointing: bool = False
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    layer_decay: Optional[float] = None


@dataclass
class ParallelismCfg:
    dist_mode: str = 'ddp'
    sharding_strategy: str = 'hybrid'
    sharded_state_dict: bool = False

    def __post_init__(self):
        assert self.dist_mode in ('ddp', 'fsdp')


@dataclass
class TaskCfg:
    dtype: Optional[str] = None
    amp: bool = True


@dataclass
class TaskTrainCfg(TaskCfg):
    model: ModelArgs = field(default_factory=ModelArgs)
    tokenizer: Optional[TokenizerCfg] = None
    image_transforms: str = "nougat"  # Can be "better", "nougat" or ""
    num_intervals: int = 100
    num_warmup_intervals: int = 5
    log_frequency: int = 100  # log every n steps
    metrics_frequency: int = 1000  # calculate train metrics every n steps
    eval_frequency: Optional[int] = None  # FIXME needs redefinition
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)
    dist: ParallelismCfg = field(default_factory=ParallelismCfg)


@dataclass
class TaskEvalCfg(TaskCfg):
    pass

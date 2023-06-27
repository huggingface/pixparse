from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OptimizationCfg:
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    learning_rate: float = 5e-4
    warmup_learning_rate: float = 0.
    weight_decay: float = .02
    clip_grad_value: Optional[float] = None
    clip_grad_mode: Optional[str] = None
    grad_accum_steps: int = 1
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None


@dataclass
class TaskTrainCfg:
    num_intervals: int = 100
    num_warmup_intervals: int = 5
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)
    dtype: Optional[str] = None
    amp: bool = True

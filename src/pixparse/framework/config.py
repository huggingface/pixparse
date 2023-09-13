from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OptimizationCfg:
    """
    This dataclass serves among others the task.opt arg set.
    """
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    learning_rate: float = 5e-4
    warmup_learning_rate: float = 0.
    weight_decay: float = .02
    eps: float = 1e-6
    clip_grad_value: Optional[float] = None
    clip_grad_mode: Optional[str] = None
    grad_accum_steps: int = 1
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    layer_decay: Optional[float] = None


@dataclass
class TaskTrainCfg:
    num_intervals: int = 100
    num_warmup_intervals: int = 5
    eval_frequency: int = 1000 
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)
    dtype: Optional[str] = None
    amp: bool = True
    model_name: str = ""
    transforms: str = "nougat" 

@dataclass
class TaskEvalCfg:
    dtype: Optional[str] = None
    amp: bool = True
    model_name: str = ""
    model_state_dict: dict = field(default_factory=dict) #FIXME move out state dict into dict of dict


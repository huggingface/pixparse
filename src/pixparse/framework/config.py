from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OptimizationCfg:
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    learning_rate: float = 5e-4
    weight_decay: float = .02
    clip_grad_value: Optional[float] = None
    clip_grad_mode: Optional[str] = None
    grad_accum_steps: int = 1
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None



from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import TaskTrainCfg, TaskEvalCfg
from .device import DeviceEnv
from .monitor import Monitor


class Task:
    def __init__(
            self,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        self.device_env = device_env
        self.monitor = monitor
    
class TaskEval(Task):  
    def __init__(
            self,
            cfg: TaskEvalCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(device_env=device_env, monitor=monitor)
    
    def setup(self, *args, **kwargs):
        pass

    def prepare_for_evaluation(self):
        pass

    def step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def end(self):
        pass

class TaskTrain(Task):
    def __init__(
            self,
            cfg: TaskTrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(device_env=device_env, monitor=monitor)

        self.num_intervals = cfg.num_intervals
        self.num_warmup_intervals = cfg.num_warmup_intervals
        self.eval_frequency = cfg.eval_frequency
        self.num_steps_per_interval = None  # uninitialized, needs dataset info
        self.start_interval = 0

        self.step = 0  # step (aka optimizer update) count
        self.batch_idx = 0  # total train batch count
        self.interval_idx = 0  # interval (aka epoch or restorable period-between-checkpoints)
        self.interval_batch_idx = 0  # batch count in current interval

        # optimization state initialized in train_setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.autocast = None

    def train_setup(self, *args, **kwargs):
        pass

    def train_interval_start(self):
        pass

    def train_interval_end(self):
        pass

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def eval_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # TODO Remove eval method from train dataclass
        pass

    def get_current_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return lr

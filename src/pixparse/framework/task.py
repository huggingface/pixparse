from typing import Any, Dict


class Task:
    def __init__(self):
        pass

    def train_setup(self, *args, **kwargs):
        pass

    def train_interval_start(self):
        pass

    def train_interval_end(self):
        pass

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def eval_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

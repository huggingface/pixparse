import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import torchvision.transforms as transforms


from pixparse.framework import TaskEvalCfg, TaskEval, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerHF, TokenizerCfg
from pixparse.data import preprocess_text_anno
from pixparse.utils import get_ocr_metrics
from pixparse.task import TaskCrullerEvalOCR, TaskCrullerEvalOCRCfg, TaskDonutEvalOCR, TaskDonutEvalOCRCfg, TaskCrullerPretrain, TaskCrullerPretrainCfg

from chug.common import LoaderBundle

_logger = logging.getLogger(__name__)

import time


class TaskFactory:
    """
    This class registers existing tasks and propagates corresponding configurations.
    """
    TASK_CONFIG_REGISTRY = {
        'cruller_eval_ocr': TaskCrullerEvalOCRCfg,
        'donut_eval_ocr': TaskDonutEvalOCRCfg,
        'cruller_pretrain': TaskCrullerPretrainCfg
    }

    TASK_CLASS_REGISTRY = {
        'cruller_eval_ocr': TaskCrullerEvalOCR,
        'donut_eval_ocr': TaskDonutEvalOCR,
        'cruller_pretrain': TaskCrullerPretrain
    }

    @classmethod
    def create_task_cfg(cls, task_name: str, args):
        task_name = task_name.lower()
        if task_name not in cls.TASK_CONFIG_REGISTRY:
            raise ValueError(f"Unknown task type: {task_name}. Available tasks are {list(cls.TASK_CONFIG_REGISTRY.keys())}")
        task_cfg_cls = cls.TASK_CONFIG_REGISTRY[task_name]
        return task_cfg_cls(**vars(args))

    @classmethod
    def create_task(cls, task_name: str, task_cfg: TaskEvalCfg, device_env: DeviceEnv, monitor: Monitor):
        task_name = task_name.lower()
        if task_name not in cls.TASK_CLASS_REGISTRY:
            raise ValueError(f"Unknown task type: {task_name}. Available tasks are {list(cls.TASK_CLASS_REGISTRY.keys())}")
        task_cls = cls.TASK_CLASS_REGISTRY[task_name]
        return task_cls(task_cfg, device_env, monitor)

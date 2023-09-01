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
from pixparse.task import (
    TaskCrullerEvalOCR,
    TaskCrullerEvalOCRCfg,
    TaskDonutEvalOCR,
    TaskDonutEvalOCRCfg,
    TaskCrullerEvalRVLCDIP,
    TaskCrullerEvalRVLCDIPCfg,
    TaskCrullerEvalCORD,
    TaskCrullerEvalCORDCfg,
    TaskCrullerPretrain,
    TaskCrullerPretrainCfg,
    TaskCrullerFinetuneRVLCDIP,
    TaskCrullerFinetuneRVLCDIPCfg,
    TaskCrullerFinetuneCORD,
    TaskCrullerFinetuneCORDCfg,
    TaskCrullerFinetuneXent,
    TaskCrullerFinetuneXentCfg,
)

from chug.common import LoaderBundle

_logger = logging.getLogger(__name__)

import time


class TaskFactory:
    """
    This class registers existing tasks and propagates corresponding configurations.
    """

    TASK_CLASS_REGISTRY = {
        "cruller_eval_ocr": (TaskCrullerEvalOCR, TaskCrullerEvalOCRCfg),
        "cruller_eval_rvlcdip": (TaskCrullerEvalRVLCDIP, TaskCrullerEvalRVLCDIPCfg),
        "cruller_eval_cord": (TaskCrullerEvalCORD, TaskCrullerEvalCORDCfg),
        "donut_eval_ocr": (TaskDonutEvalOCR, TaskDonutEvalOCRCfg),
        "cruller_pretrain": (TaskCrullerPretrain, TaskCrullerPretrainCfg),
        "cruller_finetune_rvlcdip": (
            TaskCrullerFinetuneRVLCDIP,
            TaskCrullerFinetuneRVLCDIPCfg,
        ),
        "cruller_finetune_cord": (TaskCrullerFinetuneCORD, TaskCrullerFinetuneCORDCfg),
        #"cruller_finetune_trainticket": (
        #    TaskCrullerFinetuneTrainTicket,
        #    TaskCrullerFinetuneTrainTicketCfg,
        #),
        "cruller_finetune_xent": (TaskCrullerFinetuneXent, TaskCrullerFinetuneXentCfg),
    }

    @classmethod
    def create_task(
        cls, task_name: str, task_args, device_env: DeviceEnv, monitor: Monitor
    ):
        task_name = task_name.lower()
        if task_name not in cls.TASK_CLASS_REGISTRY:
            raise ValueError(
                f"Unknown task type: {task_name}. Available tasks are {list(cls.TASK_CLASS_REGISTRY.keys())}"
            )
        task_cls = cls.TASK_CLASS_REGISTRY[task_name][0]
        task_cfg = cls.TASK_CLASS_REGISTRY[task_name][1]
        task_cfg_instance = task_cfg(**vars(task_args))
        task_cls_instance = task_cls(
            cfg=task_cfg_instance, device_env=device_env, monitor=monitor
        )
        return task_cls_instance, task_cfg_instance

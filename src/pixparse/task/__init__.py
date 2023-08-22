from .task_cruller_pretrain import TaskCrullerPretrain, TaskCrullerPretrainCfg
from .task_cruller_finetune_RVLCDIP import TaskCrullerFinetuneRVLCDIP, TaskCrullerFinetuneRVLCDIPCfg
from .task_cruller_finetune_xent import (
    TaskCrullerFinetuneXent,
    TaskCrullerFinetuneXentCfg,
)

from .task_cruller_eval_ocr import TaskCrullerEvalOCR, TaskCrullerEvalOCRCfg
from .task_donut_eval_ocr import TaskDonutEvalOCR, TaskDonutEvalOCRCfg
from .task_factory import TaskFactory

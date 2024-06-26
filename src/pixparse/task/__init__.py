from .task_cruller_pretrain import TaskCrullerPretrain, TaskCrullerPretrainCfg
from .task_cruller_finetune_RVLCDIP import TaskCrullerFinetuneRVLCDIP, TaskCrullerFinetuneRVLCDIPCfg
from .task_cruller_finetune_CORD import TaskCrullerFinetuneCORD, TaskCrullerFinetuneCORDCfg
from .task_cruller_finetune_xent import (
    TaskCrullerFinetuneXent,
    TaskCrullerFinetuneXentCfg,
)
from .task_cruller_finetune_docvqa import TaskCrullerFinetuneDOCVQA, TaskCrullerFinetuneDOCVQACfg

from .task_cruller_eval_ocr import TaskCrullerEvalOCR, TaskCrullerEvalOCRCfg
from .task_donut_eval_ocr import TaskDonutEvalOCR, TaskDonutEvalOCRCfg
from .task_cruller_eval_rvlcdip import TaskCrullerEvalRVLCDIP, TaskCrullerEvalRVLCDIPCfg
from .task_cruller_eval_cord import TaskCrullerEvalCORD, TaskCrullerEvalCORDCfg
from .task_cruller_eval_docvqa import TaskCrullerEvalDOCVQA, TaskCrullerEvalDOCVQACfg


from .task_factory import TaskFactory

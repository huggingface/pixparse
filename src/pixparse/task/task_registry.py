from copy import deepcopy

from pixparse.task import (
    TaskCrullerEvalOCR,
    TaskCrullerEvalOCRCfg,
    TaskDonutEvalOCR,
    TaskDonutEvalOCRCfg,
    TaskCrullerEvalRVLCDIP,
    TaskCrullerEvalRVLCDIPCfg,
    TaskCrullerEvalCORD,
    TaskCrullerEvalCORDCfg,
    TaskCrullerEvalDOCVQA,
    TaskCrullerEvalDOCVQACfg,
    TaskCrullerPretrain,
    TaskCrullerPretrainCfg,
    TaskCrullerFinetuneRVLCDIP,
    TaskCrullerFinetuneRVLCDIPCfg,
    TaskCrullerFinetuneCORD,
    TaskCrullerFinetuneCORDCfg,
    TaskCrullerFinetuneDOCVQA,
    TaskCrullerFinetuneDOCVQACfg,
    TaskCrullerFinetuneXent,
    TaskCrullerFinetuneXentCfg,
)

TRAIN_TASK_REGISTRY = {
    "cruller_pretrain": (TaskCrullerPretrain, TaskCrullerPretrainCfg),
    "cruller_finetune_rvlcdip": (
        TaskCrullerFinetuneRVLCDIP,
        TaskCrullerFinetuneRVLCDIPCfg,
    ),
    "cruller_finetune_cord": (TaskCrullerFinetuneCORD, TaskCrullerFinetuneCORDCfg),
    "cruller_finetune_docvqa": (TaskCrullerFinetuneDOCVQA, TaskCrullerFinetuneDOCVQACfg),
    #"cruller_finetune_trainticket": (
    #    TaskCrullerFinetuneTrainTicket,
    #    TaskCrullerFinetuneTrainTicketCfg,
    #),
    "cruller_finetune_xent": (TaskCrullerFinetuneXent, TaskCrullerFinetuneXentCfg),
}
TRAIN_TO_TASK_PAIR = {v[1]: (k, v[0]) for k, v in TRAIN_TASK_REGISTRY.items()}
TRAIN_NAME_TO_TASK = {k: v[0] for k, v in TRAIN_TASK_REGISTRY.items()}
TRAIN_NAME_TO_CFG = {k: v[1] for k, v in TRAIN_TASK_REGISTRY.items()}

EVAL_TASK_REGISTRY = {
    "cruller_eval_ocr": (TaskCrullerEvalOCR, TaskCrullerEvalOCRCfg),
    "cruller_eval_rvlcdip": (TaskCrullerEvalRVLCDIP, TaskCrullerEvalRVLCDIPCfg),
    "cruller_eval_cord": (TaskCrullerEvalCORD, TaskCrullerEvalCORDCfg),
    "cruller_eval_docvqa": (TaskCrullerEvalDOCVQA, TaskCrullerEvalDOCVQACfg),
    "donut_eval_ocr": (TaskDonutEvalOCR, TaskDonutEvalOCRCfg),
}
EVAL_CFG_TO_TASK_PAIR = {v[1]: (k, v[0]) for k, v in EVAL_TASK_REGISTRY.items()}
EVAL_NAME_TO_TASK = {k: v[0] for k, v in EVAL_TASK_REGISTRY.items()}
EVAL_NAME_TO_CFG = {k: v[1] for k, v in EVAL_TASK_REGISTRY.items()}


def get_train_task_from_cfg(cfg):
    return TRAIN_TO_TASK_PAIR[type(cfg)]


def get_train_task_cfgs():
    return deepcopy(TRAIN_NAME_TO_CFG)


def get_eval_task_from_cfg(cfg):
    return EVAL_CFG_TO_TASK_PAIR[type(cfg)]


def get_eval_task_cfgs():
    return deepcopy(EVAL_NAME_TO_CFG)

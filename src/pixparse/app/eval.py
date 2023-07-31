import logging
import os
import json
from dataclasses import dataclass, replace, field
from typing import List

import simple_parsing
from simple_parsing import ArgumentParser

import torch

from pixparse.data import DataCfg, create_loader
from pixparse.framework import (
    TaskEval,
    TaskEvalCfg,
    DeviceEnv,
    Monitor,
    evaluate,
    setup_logging,
    random_seed,
)
from pixparse.utils.s3_utils import load_checkpoint_from_s3

from pixparse.task import (
    TaskCrullerEvalOCR,
    TaskCrullerEvalOCRCfg,
    TaskDonutEvalOCR,
    TaskDonutEvalOCRCfg,
)

from chug.webdataset import create_doc_anno_pipe, create_image_text_pipe

_logger = logging.getLogger("eval")


class TaskFactory:
    TASK_CONFIG_REGISTRY = {
        'cruller_eval_ocr': TaskCrullerEvalOCRCfg,
        'donut_eval_ocr': TaskDonutEvalOCRCfg
    }

    TASK_CLASS_REGISTRY = {
        'cruller_eval_ocr': TaskCrullerEvalOCR,
        'donut_eval_ocr': TaskDonutEvalOCR
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
            raise ValueError(f"Unknown task type: {task_name}. Available eval tasks are {list(cls.TASK_CLASS_REGISTRY.keys())}")
        task_cls = cls.TASK_CLASS_REGISTRY[task_name]
        return task_cls(task_cfg, device_env, monitor)


@dataclass
class EvalCfg:
    experiment: str = ""
    output_dir: str = "./output"
    log_filename: str = "out.log"
    dataset_name: str = ""
    s3_bucket: str = ""
    checkpoint_path: str = ""
    metrics_file_path: str = ""
    task_name: str = ""
    datasets: List[str] = field(
        default_factory=lambda: ["eval"]
    )  # Identifier of dataset to be used in eval.
    seed: int = 42


def eval(
    cfg: EvalCfg,
    task: TaskEval,
    eval_loaders: dict,
):
    device_env = task.device_env

    # load wanted checkpoint

    metrics = evaluate(task, eval_loaders)
    # Do something with metrics, print them, log them, save them
    # FIXME how do we log metrics per dataset?
    with open(cfg.metrics_file_path, "w") as f:
        json.dump(metrics, f)


parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
    add_config_path_arg=True,
)
parser.add_arguments(EvalCfg, dest="eval")
parser.add_arguments(TaskEvalCfg, dest="task")
parser.add_arguments(DataCfg, dest="data")


def main():
    args = parser.parse_args()
    eval_cfg: EvalCfg = args.eval
    data_cfg: DataCfg = args.data

    # create task config

    task_cfg = TaskFactory.create_task_cfg(eval_cfg.task_name, args.task) 

    device_env = DeviceEnv()
    random_seed(
        eval_cfg.seed, rank=device_env.global_rank
    )  # Seed variability for eval?
    print(device_env)

    assert (
        eval_cfg.output_dir is not None
    ), f"output_dir is not provided. Stopping eval run."


    if device_env.is_primary():
        log_path = os.path.join(eval_cfg.output_dir, eval_cfg.log_filename)

    # Setup text logger
    setup_logging(log_path)
    monitor = Monitor(
        eval_cfg.experiment,
        output_dir=eval_cfg.output_dir,
        output_enabled=device_env.is_primary(),
    )
    
    # Check if current tasks is external model evaluation

    if eval_cfg.task_name not in ["donut"]:
        checkpoint_path = eval_cfg.checkpoint_path
        eval_cfg = replace(eval_cfg, checkpoint_path=checkpoint_path)

        # FIXME check if path is local or s3?
        if eval_cfg.s3_bucket != "":
            _logger.info("s3 bucket specified. Loading checkpoint from s3.")
            checkpoint = load_checkpoint_from_s3(
                eval_cfg.s3_bucket, eval_cfg.checkpoint_path
            )
        else:
            assert os.path.isfile(
                checkpoint_path
            ), f"Cannot find checkpoint {checkpoint_path}: File not found"

            checkpoint = torch.load(eval_cfg.checkpoint_path)
            state_dict = checkpoint["model"]


        # Create safe metrics file path

        checkpoint_name = eval_cfg.checkpoint_path.replace("/", "-").replace(".pt", "")
        metrics_file_name = f"{checkpoint_name}-{eval_cfg.dataset_name}-metrics.json"

        # bypass DDP module
        
        eval_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        task_cfg.model_state_dict = eval_state_dict
    else:
        # Get a generic name for external model on chosen dataset
        metrics_file_name = f"{eval_cfg.task_name}-{eval_cfg.dataset_name}-metrics.json"

    eval_cfg.metrics_file_path = os.path.join(eval_cfg.output_dir, metrics_file_name)

    if device_env.is_primary():
        _logger.info(task_cfg)
        _logger.info(eval_cfg)

    # Instantiate eval task

    task = TaskFactory.create_task(eval_cfg.task_name, task_cfg, device_env, monitor)

    loaders = {}
    assert data_cfg.eval is not None, f"data_cfg.eval is not set."

    # FIXME add common functionality for loader selection per task
    loaders["eval_FUNSD"] = create_loader(
        data_cfg.eval,
        is_train=False,
        image_preprocess=task.image_preprocess_eval,
        anno_preprocess=task.anno_preprocess_eval,
        create_decoder_pipe=create_image_text_pipe,
        # world_size=device_env.world_size
    )

    task.setup()

    if device_env.is_primary():
        _logger.info(task)

    eval(
        eval_cfg,
        task,
        loaders,
    )

    task.end()


if __name__ == "__main__":
    main()

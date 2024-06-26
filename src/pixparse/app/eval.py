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

from pixparse.task.task_factory import TaskFactory

from chug.webdataset import create_doc_anno_pipe, create_image_text_pipe

from collections import OrderedDict
_logger = logging.getLogger("eval")


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

    device_env = DeviceEnv()
    task, task_cfg = TaskFactory.create_task(task_name=eval_cfg.task_name, task_args=args.task, device_env=device_env, monitor=None)


    random_seed(
        eval_cfg.seed, rank=device_env.global_rank
    )  # Seed variability for eval?
    _logger.info(f"Device env is {device_env}")

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
    
    # FIXME defer load checkpoint to task?

    if eval_cfg.task_name not in ["donut_eval_ocr"]:
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
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint["model"]
        # Create safe metrics file path

        checkpoint_name = eval_cfg.checkpoint_path.replace("/", "_").replace(".pt", "")
        metrics_file_name = f"{checkpoint_name}-{eval_cfg.dataset_name}-metrics.json"

        # bypass DDP module
        
        eval_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        task.resume_state_dict = eval_state_dict
    else:
        # Get a generic name for external model on chosen dataset
        metrics_file_name = f"{eval_cfg.task_name}-{eval_cfg.dataset_name}-metrics.json"

    eval_cfg.metrics_file_path = os.path.join(eval_cfg.output_dir, metrics_file_name)

    if device_env.is_primary():
        _logger.info(task_cfg)
        _logger.info(eval_cfg)



    loaders = {}
    assert data_cfg.eval is not None, f"data_cfg.eval is not set."

    # FIXME add common functionality for loader selection per task
    loaders["eval"] = create_loader(
        data_cfg.eval,
        is_train=False,
        collate_fn=task.collate_fn,
        image_preprocess=task.image_preprocess_eval,
        anno_preprocess=task.anno_preprocess_eval,
        image_fmt=task_cfg.model.image_encoder.image_fmt,
        world_size=device_env.world_size,
        local_rank=device_env.local_rank,
        create_decoder_pipe=create_image_text_pipe, # TODO abstract away type of decoder needed
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

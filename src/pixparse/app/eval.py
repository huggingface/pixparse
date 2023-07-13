import logging
import os
import json
from dataclasses import dataclass, replace, field
from datetime import datetime
from typing import Optional, List

import simple_parsing
from simple_parsing import ArgumentParser

import torch

from pixparse.data import DataCfg, create_loader
from pixparse.framework import (
    DeviceEnv,
    Monitor,
    train_one_interval,
    evaluate,
    setup_logging,
    random_seed,
)
from pixparse.utils.name_utils import clean_name
from pixparse.utils.s3_utils import load_checkpoint_from_s3

from pixparse.task import (
    TaskCrullerPretrain,
    TaskCrullerPretrainCfg,
    TaskCrullerEvalOCR,
    TaskCrullerEvalOCRCfg,
)

from chug.webdataset import create_doc_anno_pipe, create_image_text_pipe

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
    datasets: List[str] = field(
        default_factory=lambda: ["eval"]
    )  # Identifier of dataset to be used in eval.
    seed: int = 42


def eval(
    cfg: EvalCfg,
    task: TaskCrullerEvalOCR,  # FIXME define common functionality in interface
    eval_loaders: dict,
):
    device_env = task.device_env

    # load wanted checkpoint

    metrics = evaluate(task, eval_loaders)
    # Do something with metrics, print them, log them, save them
    # FIXME how do we log metrics per dataset? 
    with open(cfg.metrics_file_path , "w") as f:
        json.dump(metrics, f)


parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
    add_config_path_arg=True,
)
parser.add_arguments(EvalCfg, dest="eval")
parser.add_arguments(TaskCrullerEvalOCRCfg, dest="task")
parser.add_arguments(DataCfg, dest="data")


def main():
    args = parser.parse_args()
    eval_cfg: EvalCfg = args.eval
    task_cfg: TaskCrullerEvalOCRCfg = args.task
    data_cfg: DataCfg = args.data

    device_env = DeviceEnv()
    random_seed(
        eval_cfg.seed, rank=device_env.global_rank
    )  # Seed variability for eval?
    print(device_env)

    # assert eval_cfg.experiment is not None, f"experiment is not provided. Stopping eval run."
    assert (
        eval_cfg.output_dir is not None
    ), f"output_dir is not provided. Stopping eval run."

    # experiment_path = os.path.join(eval_cfg.output_dir, eval_cfg.experiment)
    # assert os.path.isdir(
    #    experiment_path
    # ), f"Cannot find experiment location {experiment_path}: No such directory."
    if device_env.is_primary():
        log_path = os.path.join(eval_cfg.output_dir, eval_cfg.log_filename)

    # Setup text logger
    setup_logging(log_path)
    monitor = Monitor(
        eval_cfg.experiment,
        output_dir=eval_cfg.output_dir,
        output_enabled=device_env.is_primary(),
    )

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

    # Create safe metrics file path 

    checkpoint_name = eval_cfg.checkpoint_path.replace('/', '-').replace('.pt', '')
    metrics_file_name = f"{checkpoint_name}-{eval_cfg.dataset_name}-metrics.json"    
    eval_cfg.metrics_file_path = os.path.join(eval_cfg.output_dir, metrics_file_name)


    state_dict = checkpoint["model"]
    # bypass DDP module
    eval_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    task_cfg.model_state_dict = eval_state_dict

    if device_env.is_primary():
        _logger.info(task_cfg)
        _logger.info(eval_cfg)

    # Instantiate eval task

    task = TaskCrullerEvalOCR(
        task_cfg,
        device_env,
        monitor,
    )

    loaders = {}
    assert data_cfg.eval is not None, f"data_cfg.eval is not set."
    loaders["eval_FUNSD"] = create_loader(
        data_cfg.eval,
        is_train=False,
        image_preprocess=task.image_preprocess_eval,
        anno_preprocess=task.anno_preprocess_eval,
        create_decoder_pipe=create_image_text_pipe,
        # world_size=device_env.world_size
    )
    """
    loaders["eval_FUNSD"] = create_loader(
        data_cfg.eval,
        is_train=False,
        image_preprocess=task.image_preprocess_eval,
        anno_preprocess=task.anno_preprocess_eval,
        create_decoder_pipe=create_image_text_pipe,
        # world_size=device_env.world_size
    )
    """
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

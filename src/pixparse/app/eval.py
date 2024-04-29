import logging
import os
import json
from dataclasses import dataclass, replace, field
from typing import List

import simple_parsing
from simple_parsing import ArgumentParser, subgroups
from simple_parsing.helpers import Serializable

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
from pixparse.utils import get_selected_non_default_args
from pixparse.task import get_eval_task_from_cfg, get_eval_task_cfgs

from chug.webdataset import create_doc_anno_pipe, create_image_text_pipe

from collections import OrderedDict
_logger = logging.getLogger("eval")


@dataclass
class EvalCfg(Serializable):
    eval_data: DataCfg
    task: TaskEvalCfg = subgroups(get_eval_task_cfgs(), default='cruller_eval_docvqa')

    experiment: str = ""
    output_dir: str = "./output"
    log_filename: str = "out_new.log"
    dataset_name: str = ""
    s3_bucket: str = ""
    checkpoint_path: str = ""
    metrics_file_path: str = ""
    task_name: str = ""
    #datasets: List[str] = field(
    #    default_factory=lambda: ["eval"]
    #)  # Identifier of dataset to be used in eval.
    seed: int = 42


def eval(
    cfg: EvalCfg,
    task: TaskEval,
    eval_loaders: dict,
):
    
    device_env = task.device_env

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
parser.add_arguments(EvalCfg, dest='cfg')


def main():
    args = parser.parse_args()
    eval_cfg: EvalCfg = args.cfg
    print(eval_cfg.dumps_yaml())

    device_env = DeviceEnv()
    task_name, task_cls = get_eval_task_from_cfg(eval_cfg.task)
    # task, task_cfg = TaskFactory.create_task(task_name=eval_cfg.task_name,
    #                                         task_args=args.task, device_env=device_env, monitor=None)

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
    selected_args = ['task', 'checkpoint_path']
    rename_map = {'task': 'cfg'}
    selected_non_default_args = get_selected_non_default_args(eval_cfg, selected_args, rename_map)

    task = task_cls(
        **selected_non_default_args,
        device_env=device_env,
        monitor=monitor,
    )

    checkpoint_name = eval_cfg.checkpoint_path.replace("/", "_").replace(".pt", "")
    metrics_file_name = f"{checkpoint_name}-{eval_cfg.dataset_name}-metrics.json"
    eval_cfg.metrics_file_path = os.path.join(eval_cfg.output_dir, metrics_file_name)

    if device_env.is_primary():
        _logger.info(eval_cfg)

    loaders = {}

    # TODO add common functionality for loader selection per task
    loaders["eval"] = create_loader(
        eval_cfg.eval_data,
        is_train=False,
        collate_fn=task.collate_fn,
        image_preprocess=task.image_preprocess_eval,
        anno_preprocess=task.anno_preprocess_eval,
        image_fmt=task.image_input_cfg.image_fmt,
        world_size=device_env.world_size,
        global_rank=device_env.global_rank,
        create_decoder_pipe=create_image_text_pipe,  # TODO abstract away type of decoder needed
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

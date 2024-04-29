import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, Optional, Union

import simple_parsing
from simple_parsing import ArgumentParser, subgroups
from simple_parsing.helpers import Serializable

import torch

from pixparse.data import DataCfg, create_loader
from pixparse.framework import (DeviceEnv, Monitor, train_one_interval, evaluate, setup_logging, random_seed,
                                TaskTrain, TaskTrainCfg)
from pixparse.utils import clean_name, load_checkpoint, get_selected_non_default_args
from pixparse.task import get_train_task_from_cfg, get_train_task_cfgs

from chug.common import LoaderBundle
from chug.webdataset import create_doc_anno_pipe

_logger = logging.getLogger('train')


@dataclass
class TrainCfg(Serializable):
    train_data: DataCfg
    eval_data: Optional[DataCfg] = None
    task: TaskTrainCfg = subgroups(get_train_task_cfgs(), default='cruller_pretrain')

    experiment: Optional[str] = None  # experiment name, auto-generated if None or required?
    output_dir: str = './output'
    log_filename: str = 'out.log'
    resume: Optional[str] = ""  # resume checkpoint path w/ full model + optimizer state
    checkpoint_path: Optional[str] = "" # general checkpoint path to initialize model with
    output_checkpoint_dir: Optional[str] = None  # default output_dir/checkpoints
    seed: int = 42

    wandb: bool = False
    wandb_project: str = 'unknown'
    tensorboard: bool = False
    log_eval_data: bool = False


def train(
        cfg: TrainCfg,
        task: TaskTrain,
        loaders: Dict[str, LoaderBundle],
):
    device_env = task.device_env
    train_loader = loaders['train']
    for i in range(task.interval_idx, task.num_intervals):
        # FIXME flatten interval loop to have one eval point
        #  i.e step intervals vs epoch intervals handled similarly?
        train_loader.set_interval(i)
        train_one_interval(
            task,
            train_loader,
            cfg,
            interval=i
        )

        # save checkpoint
        if device_env.is_primary():
            checkpoint_dir = os.path.join(cfg.output_checkpoint_dir, cfg.experiment)
            os.makedirs(checkpoint_dir, exist_ok=True)
            if i in [1, 9, 29, 39, 40, 49, 59, 69, 79, 89]:
                torch.save(task.state_dict(), os.path.join(checkpoint_dir, f'checkpoint-{i}.pt'))
        
"""
def load_checkpoint_cases(train_cfg, task):
    # ----- Model resuming from checkpoint (non-fsdp) -----
    if train_cfg.resume:
        if not train_cfg.experiment:
            raise ValueError(
                f"resume set to {train_cfg.resume} and experiment directory not found at {train_cfg.experiment}.")
        # FIXME add 'resume_latest' mode that scans experiment path for latest checkpoint
        raise NotImplementedError("resume is not implemented yet. ")
        if checkpoint_path.startswith('s3'):
            _logger.info("s3 bucket specified. Loading checkpoint from s3 for resuming.")
        else:
            _logger.info("Loading checkpoint from local path for resuming.")
        checkpoint = load_checkpoint(checkpoint_path)
        task.load_state_dict(checkpoint, restore_optimizer_state=True)
    # ----- Model being finetuned from checkpoint (non-fsdp) ----
    elif train_cfg.checkpoint_path:
        # FIXME improve naming/separation between resume and finetune, assert both can't be used at the same time!
        checkpoint_path = train_cfg.checkpoint_path
        if checkpoint_path.startswith('s3'):
            _logger.info("s3 bucket specified. Loading checkpoint from s3 for finetuning.")
        else:
            _logger.info("Loading checkpoint from local path for finetuning.")
        checkpoint = load_checkpoint(checkpoint_path)
        task.load_state_dict(
            checkpoint,
            restore_optimizer_state=False,
            restore_scheduler_state=False,
            )
"""

parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
    add_config_path_arg=True,
)
parser.add_arguments(TrainCfg, dest='cfg')


def main():
    args = parser.parse_args()
    train_cfg: TrainCfg = args.cfg

    device_env = DeviceEnv()
    random_seed(train_cfg.seed, rank=device_env.global_rank)
    _logger.info(f"Device env is {device_env}")

    print(train_cfg.dumps_yaml())
    task_name, task_cls = get_train_task_from_cfg(train_cfg.task)
    # get the name of the experiments
    if train_cfg.experiment is None:
        model_name_safe = clean_name(train_cfg.task.model.name)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        if device_env.world_size > 1:
            # sync date_str from master to all ranks
            date_str = device_env.broadcast_object(date_str)
        experiment = '-'.join([
            date_str,
            f"task_{task_name}",
            f"model_{model_name_safe}",
            f"lr_{'{:.1e}'.format(train_cfg.task.opt.learning_rate)}",
            f"b_{train_cfg.train_data.batch_size}",
            f"intervals_{train_cfg.task.num_intervals}",
            # TODO make completion of exp name derived from essential hparams
        ])
        train_cfg = replace(train_cfg, experiment=experiment)

    resume_latest = False  # train_cfg.resume == 'latest'
    experiment_path = os.path.join(train_cfg.output_dir, train_cfg.experiment)
    log_path = None
    if device_env.is_primary():
        os.makedirs(experiment_path, exist_ok=True)
        log_path = os.path.join(experiment_path, train_cfg.log_filename)
        if os.path.exists(log_path) and not resume_latest:
            _logger.error(
                "Error. Experiment already exists. Use --experiment {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    setup_logging(log_path)
    monitor = Monitor(
        train_cfg.experiment,
        output_dir=experiment_path,
        wandb=train_cfg.wandb,
        wandb_project=train_cfg.wandb_project,
        tensorboard=train_cfg.tensorboard,
        output_enabled=device_env.is_primary(),
    )

    selected_args = ['task', 'resume', 'checkpoint_path']
    rename_map = {'task': 'cfg'}
    selected_non_default_args = get_selected_non_default_args(train_cfg, selected_args, rename_map)
    task = task_cls(
        **selected_non_default_args,
        device_env=device_env,
        monitor=monitor,
    )

    # FIXME Move this functionality to task_cls instance init so that all weight init is in init
    # load_checkpoint_cases(train_cfg, task)

    output_checkpoint_dir = train_cfg.output_checkpoint_dir or os.path.join(experiment_path, 'checkpoints')
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    train_cfg = replace(train_cfg, output_checkpoint_dir=output_checkpoint_dir)
    if device_env.is_primary():
        _logger.info(train_cfg)

    loaders = {}
    assert train_cfg.train_data is not None, f"Train dataset (train_cfg.train_data) must be set."
    loaders['train'] = create_loader(
        train_cfg.train_data,
        is_train=True,
        collate_fn=task.collate_fn,
        image_preprocess=task.image_preprocess_train,
        anno_preprocess=task.anno_preprocess_train,
        image_fmt=task.image_input_cfg.image_fmt,
        world_size=device_env.world_size,
        global_rank=device_env.global_rank,
        create_decoder_pipe=create_doc_anno_pipe,  # TODO abstract away type of decoder needed
    )
    task.setup(
        num_batches_per_interval=loaders['train'].num_batches,
    )

    if device_env.is_primary():
        _logger.info(task)

    train(
        train_cfg,
        task,
        loaders,
    )


if __name__ == '__main__':
    main()

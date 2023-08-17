import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

import simple_parsing
from simple_parsing import ArgumentParser

import torch

from pixparse.data import DataCfg, create_loader
from pixparse.framework import DeviceEnv, Monitor, train_one_interval, evaluate, setup_logging, random_seed, TaskTrain, TaskTrainCfg
from pixparse.utils.name_utils import clean_name
from pixparse.utils.s3_utils import load_checkpoint_from_s3
from pixparse.task import TaskFactory

from chug.webdataset import create_doc_anno_pipe
_logger = logging.getLogger('train')


@dataclass
class TrainCfg:
    experiment: Optional[str] = None  # experiment name, auto-generated if None or required?
    output_dir: str = './output'
    log_filename: str = 'out.log'
    s3_bucket: str = ""
    resume: bool = False
    checkpoint_path: str = ""
    output_checkpoint_dir: Optional[str] = None  # default output_dir/checkpoints
    seed: int = 42

    # TODO
    # resume -- resume experiment from location, mode, etc
    task_name: str = "cruller_pretrain"

    wandb: bool = False
    wandb_project: str = 'unknown'

    tensorboard: bool = False
    log_eval_data: bool = False


def train(
        cfg: TrainCfg,
        task: TaskTrain,
        loaders,
):
    device_env = task.device_env
    for i in range(task.start_interval, task.num_intervals):
        # FIXME flatten interval loop to have one eval point
        #  i.e step intervals vs epoch intervals handled similarly?
        train_one_interval(
            task,
            loaders['train'],
        )

        # save checkpoint
        if device_env.is_primary():
            checkpoint_dir =  os.path.join(cfg.output_checkpoint_dir, cfg.experiment)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(task.model.state_dict(), os.path.join(checkpoint_dir, f'checkpoint-{i}.pt'))


parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
    add_config_path_arg=True,
)
parser.add_arguments(TrainCfg, dest='train')
parser.add_arguments(TaskTrainCfg, dest='task')
parser.add_arguments(DataCfg, dest='data')


def main():
    args = parser.parse_args()
    train_cfg: TrainCfg = args.train
    data_cfg: DataCfg = args.data

    device_env = DeviceEnv()
    task, task_cfg = TaskFactory.create_task(task_name=train_cfg.task_name, task_args=args.task, device_env=device_env, monitor=None)
    
    random_seed(train_cfg.seed, rank=device_env.global_rank)
    _logger.info(f"Device env is {device_env}")

    # get the name of the experiments
    if train_cfg.experiment is None:
        model_name_safe = clean_name(task_cfg.model_name)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        if device_env.world_size > 1:
            # sync date_str from master to all ranks
            date_str = device_env.broadcast_object(date_str)
        experiment = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{task_cfg.opt.learning_rate}",
            f"b_{data_cfg.train.batch_size}",
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
    task.monitor = Monitor(
        train_cfg.experiment,
        output_dir=experiment_path,
        wandb=train_cfg.wandb,
        wandb_project=train_cfg.wandb_project,
        tensorboard=train_cfg.tensorboard,
        output_enabled=device_env.is_primary(),
    )
    
    # ----- Model resuming from checkpoint -----
    # FIXME make optional for resume. Task needs to have an attribute state_dict
    if train_cfg.resume:

        checkpoint_path = train_cfg.checkpoint_path
        train_cfg = replace(train_cfg, checkpoint_path=checkpoint_path)

        # FIXME check if path is local or s3?
        if train_cfg.s3_bucket != "":
            _logger.info("s3 bucket specified. Loading checkpoint from s3.")
            checkpoint = load_checkpoint_from_s3(
                train_cfg.s3_bucket, train_cfg.checkpoint_path
            )
        else:
            assert os.path.isfile(
                checkpoint_path
            ), f"Cannot find checkpoint {checkpoint_path}: File not found"

            checkpoint = torch.load(train_cfg.checkpoint_path)
        state_dict = checkpoint["model"]
        task.state_dict = state_dict
        task.resume = True

    # ------------------------------------------

    output_checkpoint_dir = train_cfg.output_checkpoint_dir or os.path.join(experiment_path, 'checkpoints')
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    train_cfg = replace(train_cfg, output_checkpoint_dir=output_checkpoint_dir)
    if device_env.is_primary():
        _logger.info(task_cfg)
        _logger.info(train_cfg)

    loaders = {}
    assert (data_cfg.train is not None) or (data_cfg.eval is not None), f"Neither data_cfg.train nor data_cfg.eval are set."
    if data_cfg.train is not None:
        
        loaders['train'] = create_loader(
        data_cfg.train,
        is_train=True,
        collate_fn=task.collate_fn,
        image_preprocess=task.image_preprocess_train,
        anno_preprocess=task.anno_preprocess_train,
        image_fmt=task_cfg.model.image_encoder.image_fmt,
        world_size=device_env.world_size,
        local_rank=device_env.local_rank,
        create_decoder_pipe=create_doc_anno_pipe, # TODO abstract away type of decoder needed
    )
    task.train_setup(
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

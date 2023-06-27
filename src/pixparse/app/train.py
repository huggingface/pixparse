import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

import simple_parsing
from simple_parsing import ArgumentParser

import torch

from pixparse.data import DataCfg, create_loader
from pixparse.framework import DeviceEnv, Monitor, train_one_interval, evaluate, setup_logging, random_seed
from pixparse.task import TaskCrullerPretrain, TaskCrullerPretrainCfg

_logger = logging.getLogger('train')


@dataclass
class TrainCfg:
    experiment: Optional[str] = None  # experiment name, auto-generated if None or required?
    output_dir: str = './output'
    log_filename: str = 'out.log'
    checkpoint_dir: Optional[str] = None  # default output_dir/checkpoints
    seed: int = 42

    # TODO
    # resume -- resume experiment from location, mode, etc

    wandb: bool = False
    wandb_project: str = 'unknown'

    tensorboard: bool = False


def train(
        cfg: TrainCfg,
        task: TaskCrullerPretrain,  # FIXME define common functionality in interface
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

        if 'eval' in loaders:
            metrics = evaluate(
                task,
                loaders['eval']
            )
        else:
            metrics = {}

        # save checkpoint
        # checkpointer.save(task, metrics, interval)
        if device_env.is_primary():
            torch.save(task.state_dict(), os.path.join(cfg.checkpoint_dir, f'checkpoint-{i}.pt'))


parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
)
parser.add_argument("--foo", type=int, default=123, help="foo help")
parser.add_arguments(TrainCfg, dest='train')
parser.add_arguments(TaskCrullerPretrainCfg, dest='task')
parser.add_arguments(DataCfg, dest='data')


def main():
    args = parser.parse_args()
    train_cfg: TrainCfg = args.train
    task_cfg: TaskCrullerPretrainCfg = args.task

    device_env = DeviceEnv()
    random_seed(train_cfg.seed, rank=device_env.global_rank)
    print(device_env)

    # get the name of the experiments
    if train_cfg.experiment is None:
        model_name_safe = task_cfg.model.name.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if device_env.world_size > 1:
            # sync date_str from master to all ranks
            date_str = device_env.broadcast_object(date_str)
        experiment = '-'.join([
            date_str,
            f"model_{model_name_safe}",
        ])
        train_cfg = replace(train_cfg, experiment=experiment)

    resume_latest = False  # train_cfg.resume == 'latest'
    experiment_path = os.path.join(train_cfg.output_dir, train_cfg.experiment)
    log_path = None
    if device_env.is_primary():
        os.makedirs(experiment_path, exist_ok=True)
        log_path = os.path.join(experiment_path, train_cfg.log_filename)
        if os.path.exists(log_path) and not resume_latest:
            print(
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

    checkpoint_dir = train_cfg.checkpoint_dir or os.path.join(experiment_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_cfg = replace(train_cfg, checkpoint_dir=checkpoint_dir)
    if device_env.is_primary():
        _logger.info(train_cfg)

    task = TaskCrullerPretrain(
        task_cfg,
        device_env,
        monitor,
    )

    data_cfg: DataCfg = args.data
    loaders = {}
    loaders['train'] = create_loader(
        data_cfg.train,
        is_train=True,
        image_preprocess=task.image_preprocess_train,
        anno_preprocess=task.anno_preprocess_train,
        image_fmt=task_cfg.model.image_encoder.image_fmt,
        world_size=device_env.world_size,
    )
    if data_cfg.eval is not None:
        loaders['eval'] = create_loader(
            data_cfg.eval,
            is_train=False,
            image_preprocess=task.image_preprocess_eval,
            anno_preprocess=task.anno_preprocess_eval,
            #world_size=device_env.world_size
        )

    task.train_setup(
        num_steps_per_interval=loaders['train'].num_batches,
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

from dataclasses import dataclass
from typing import Optional

import simple_parsing
from simple_parsing import ArgumentParser

from pixparse.data import DataCfg, create_loader
from pixparse.framework import DeviceEnv, Task, train_one_interval, evaluate
from pixparse.task import TaskCrullerPretrain, TaskCrullerPretrainConfig


@dataclass
class TrainCfg:
    experiment: Optional[str] = None  # experiment name, auto-generated if None or required?
    output_dir: str = './output'
    checkpoint_dir: Optional[str] = None  # default output_dir/checkpoints
    # TODO
    # resume -- resume experiment from location, mode, etc
    # wandb -- wandb config
    # tensorboard -- tensorboard config


def train(
        cfg: TrainCfg,
        task: Task,
        loaders,
):
    intervals = 100
    for i in range(intervals):
        # FIXME flatten interval loop to have one eval point
        #  i.e step intervals vs epoch intervals handled similarly?
        train_one_interval(
            task,
            loaders['train'],
        )

        metrics = evaluate(
            task,
            loaders['eval']
        )

        # save checkpoint
        # checkpointer.save(task, metrics, interval)


parser = ArgumentParser(
    add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
    argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
)
parser.add_argument("--foo", type=int, default=123, help="foo help")
parser.add_arguments(TrainCfg, dest='train')
parser.add_arguments(TaskCrullerPretrainConfig, dest='task')
parser.add_arguments(DataCfg, dest='data')


def main():
    args = parser.parse_args()
    train_cfg = args.train
    task_cfg = args.task

    device_env = DeviceEnv()
    print(device_env)

    task = TaskCrullerPretrain(task_cfg, device_env)

    data_cfg: DataCfg = args.data
    loaders = {}
    loaders['train'] = create_loader(
        data_cfg.train,
        is_train=True,
        image_preprocess=task.image_preprocess_train,
        anno_preprocess=task.anno_preprocess_train,
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
        num_intervals=10,
        num_warmup_intervals=1,
        num_steps_per_interval=loaders['train'].num_batches,
    )
    if device_env.is_primary():
        print(task)

    train(
        train_cfg,
        task,
        loaders,
    )


if __name__ == '__main__':
    main()

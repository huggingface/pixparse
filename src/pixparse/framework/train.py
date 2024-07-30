from .task import TaskTrain
import torch
import os


def train_one_interval(
        task: TaskTrain,
        loader,
        cfg,
        interval: int
):
    task.interval_start()

    for batch_index, sample in enumerate(loader.loader):
        output, loss = task.step(sample)
        task.after_step(sample, output, loss)
        if batch_index in [1152, 2303]: # [1535, 3071]:       
            checkpoint_dir = os.path.join(cfg.output_checkpoint_dir, cfg.experiment)
            os.makedirs(checkpoint_dir, exist_ok=True)
            #if i in [1, 5, 10, 20, 29, 40, 70, 99, 120, 150, 170, 199, 220, 250, 270, 299]:
            torch.save(task.state_dict(), os.path.join(checkpoint_dir, f'checkpoint-{interval}-{batch_index}.pt'))

    task.interval_end()

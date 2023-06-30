from .task import TaskTrain
import torch
import os

def train_one_interval(
        cfg,
        task: TaskTrain,
        interval_index, 
        loader,
):
    task.train_interval_start()

    for i, sample in enumerate(loader.loader):
        task.train_step(sample)
        # FIXME This is a debug save to check the evolution of checkpoints
        #if i%500 == 0:
        #    torch.save(task.state_dict(), os.path.join(os.path.join(cfg.output_dir, cfg.experiment), f'checkpoint-{interval_index}-{i}.pt'))

    task.train_interval_end()

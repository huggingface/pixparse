from .task import TaskTrain
import torch
import os


def train_one_interval(
        task: TaskTrain,
        loader,
):
    task.interval_start()

    for i, sample in enumerate(loader.loader):
        output, loss = task.step(sample)
        task.after_step(sample, output, loss)

    task.interval_end()

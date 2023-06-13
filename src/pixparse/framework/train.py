from .task import Task


def train_one_interval(
        task: Task,
        loader,
):
    task.train_interval_start()

    for i, sample in enumerate(loader.loader):
        task.train_step(sample)

    task.train_interval_end()

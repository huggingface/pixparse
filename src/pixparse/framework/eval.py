from .task import TaskEval


def evaluate(task: TaskEval, loaders):
    # loaders a mapping? or tuple / dataclass with evaluation loader + attributes to specify what types of eval tasks are valid?
    # loaders_and_tasks is collated container of an eval dataset loader
    # + list of eval tasks compat with each loader
    metrics = dict()
    authorized_loaders = task.prepare_for_evaluation(loaders)
    # FIXME (Pablo) not sure if I understand this correctly,
    #  are tasks in loader_and tasks -training- tasks? or other eval tasks?
    for key, loader in authorized_loaders.items():
        metrics[key] = dict()
        for index_batch, sample in enumerate(loader.loader):
            metrics[key][index_batch] = task.step(sample)

        if hasattr(task, 'average_metrics'):
            # This is the end/finalize method to aggregate metrics
            averaged_metrics = task.average_metrics(metrics[key])
            metrics[key] = {}
            metrics[key]["average"] = averaged_metrics
    return metrics
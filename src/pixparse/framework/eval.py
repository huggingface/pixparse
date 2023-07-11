from .task import TaskEval


def evaluate(task: TaskEval, loaders):
    # loaders a mapping? or tuple / dataclass with evaluation loader + attributes to specify what types of eval tasks are valid?
    # loaders_and_tasks is collated container of an eval dataset loader
    # + list of eval tasks compat with each loader
    metrics = dict()
    authorized_loaders = task.prepare_for_evaluation(loaders)
    # FIXME (Pablo) not sure if I understand this correctly,
    #  are tasks in loader_and tasks -training- tasks? or other eval tasks?
    # If they are train tasks, it means each train task must have en eval_step
    # Which feels less general-purpose
    for key, loader in authorized_loaders.items():
        metrics[key] = dict()
        for index_batch, sample in enumerate(loader.loader):
            metrics[key][index_batch] = task.step(sample)

        # for t in tasks:
        # aggregate metrics for each loader
    return metrics


# end / finalize, etc during eval some metrics can be computed per step
# and simply accumulated/averaged so the end result is used,
#  but often, eval metrics need to be computed after seeing
# all eval samples for a given dataset, caching outputs, etc.
# The end/finalize fn would calculate final metrics,
# clear any cached outputs, and return the metrics dict

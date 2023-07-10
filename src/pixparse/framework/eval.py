from .task import TaskTrain


def evaluate(
        task: TaskTrain,  # FIXME eval task?
        loaders):
        # loaders a mapping? or tuple / dataclass with evaluation loader + attributes to specify what types of eval tasks are valid?
        # loaders_and_tasks is collated container of an eval dataset loader 
        # + list of eval tasks compat with each loader
        loaders_and_tasks = task.prepare_evaluation(loaders) 
        for loader, tasks in loaders_and_tasks:
                for sample in loader:
                        for t in tasks:
                                t.eval_step(sample)
   
                for t in tasks:
                        metrics[loader.name].updated(t.end())  
                        # end / finalize, etc during eval some metrics can be computed per step 
                        # and simply accumulated/averaged so the end result is used,
                        #  but often, eval metrics need to be computed after seeing 
                        # all eval samples for a given dataset, caching outputs, etc.
                        # The end/finalize fn would calculate final metrics, 
                        # clear any cached outputs, and return the metrics dict

from .task import TaskTrain
import torch
import os
from pixparse.utils.s3_mover import S3Mover
import torch.distributed as dist

def train_one_interval(
        task: TaskTrain,
        loader,
        interval: int
):
    task.train_interval_start()

    for i, sample in enumerate(loader.loader):
        task.train_step(sample)
        checkpoint_dir = os.path.join(task.cfg.output_checkpoint_dir, task.cfg.experiment)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if task.device_env.is_primary():
            s3_mover = S3Mover(local_path=checkpoint_dir,
                s3_path=os.path.join("s3://pixparse-exps/", task.cfg.experiment),
                remove_after_upload=True,
                s5cmd_numworkers=12,
                s5cmd_concurrency=10,
                s5cmd_path="/fsx/pablo/anaconda3/envs/pp201timm098/bin/s5cmd",
                )
            s3_mover.update()
            world_process_group = task.device_env.get_world_process_group()
            s3_mover.distributed_wait_for_completion(dist, world_process_group)
            checkpoint = {
                'interval': i,
                'state_dict': task.model.state_dict(),
                'optimizer': task.optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint-{interval}-{i}.pt'))
            s3_mover.start_uploading()

    task.train_interval_end()

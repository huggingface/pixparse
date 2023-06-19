""" Distributed training/validation utils

"""
import os
from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Union, Optional, List, Tuple

import torch
import torch.distributed as dist


def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class DeviceEnvType(Enum):
    """ Device Environment Types
    """
    CPU = "cpu"
    CUDA = "cuda"
    XLA = "xla"


@dataclass
class DeviceEnv:
    init_device_type: InitVar[Optional[str]] = None
    init_device_index: InitVar[Optional[int]] = None
    init_dist_backend: InitVar[str] = 'nccl'
    init_dist_url: InitVar[str] = 'env://'

    device: torch.device = field(init=False)  # set from device_type + device_index or post_init logic
    world_size: Optional[int] = None  # set by post_init from env when None
    local_rank: Optional[int] = None  # set by post_init from env when None
    global_rank: Optional[int] = None  # set by post_init from env when None

    def is_global_primary(self):
        return self.global_rank == 0

    def is_local_primary(self):
        return self.local_rank == 0

    def is_primary(self, local=False):
        return self.is_local_primary() if local else self.is_global_primary()

    def __post_init__(
            self,
            init_device_type: Optional[str],
            init_device_index: Optional[int],
            init_dist_backend: str,
            init_dist_url: str,
    ):
        # FIXME support different device types, just using cuda to start
        assert torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        init_local_rank, init_global_rank, init_world_size = world_info_from_env()
        if init_world_size > 1:
            # setup distributed
            assert init_device_index is None
            self.local_rank = int(init_local_rank)
            is_slurm = 'SLURM_PROCID' in os.environ
            if 'SLURM_PROCID' in os.environ:
                # os.environ['LOCAL_RANK'] = str(init_local_rank)
                # os.environ['RANK'] = str(init_global_rank)
                # os.environ['WORLD_SIZE'] = str(init_world_size)
                torch.distributed.init_process_group(
                    backend=init_dist_backend,
                    init_method=init_dist_url,
                    world_size=init_world_size,
                    rank=init_global_rank,
                )
            else:
                torch.distributed.init_process_group(
                    backend=init_dist_backend,
                    init_method=init_dist_url,
                )

            self.world_size = torch.distributed.get_world_size()
            self.global_rank = torch.distributed.get_rank()
            if is_slurm:
                assert self.world_size == init_world_size
                assert self.global_rank == init_global_rank

            self.device = torch.device('cuda:%d' % self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cuda' if init_device_index is None else f'cuda:{init_device_index}')
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0

    def broadcast_object(self, obj, src=0):
        # broadcast a pickle-able python object from rank-0 to all ranks
        if self.global_rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]

    def all_gather_object(self, obj, dst=0):
        # gather a pickle-able python object across all ranks
        objects = [None for _ in range(self.world_size)]
        dist.all_gather_object(objects, obj)
        return objects

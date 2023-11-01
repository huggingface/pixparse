import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy


def fsdp_wrap_model(
        model,
        layers,
        device,
        cpu_offload=False,
        limit_all_gathers=True,
        mixed_precision_dtype=None,
):
    mixed_precision = None
    if mixed_precision_dtype is not None:
        mixed_precision = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            #reduce_dtype=,
        )

    wrapper_kwargs = dict(
        mixed_precision=mixed_precision,
        limit_all_gathers=limit_all_gathers,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        auto_wrap_policy=ModuleWrapPolicy(layers),
        use_orig_params=True,
        sync_module_states=True,
        device_id=device,
    )
    wrapped_model = FSDP(model, **wrapper_kwargs)
    return wrapped_model

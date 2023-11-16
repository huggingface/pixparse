import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy


def fsdp_wrap_model(
        model,
        layers,
        device,
        mixed_precision_dtype=None,
        reduce_full_precision=False,
        sharding_strategy='full',
        limit_all_gathers=True,
        cpu_offload=False,
):
    mixed_precision = None
    if mixed_precision_dtype is not None:
        reduce_dtype = torch.float32 if reduce_full_precision else None
        mixed_precision = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=reduce_dtype,
        )

    assert sharding_strategy in ('full', 'hybrid')
    sharding_strategy = ShardingStrategy.FULL_SHARD if sharding_strategy == 'full' else ShardingStrategy.HYBRID_SHARD

    wrapper_kwargs = dict(
        sharding_strategy=sharding_strategy,
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

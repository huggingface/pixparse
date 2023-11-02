import abc
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch

import timm
import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from pixparse.utils import load_checkpoint
from .config import TaskTrainCfg, TaskEvalCfg
from .device import DeviceEnv
from .fsdp import fsdp_wrap_model
from .monitor import Monitor


_logger = logging.getLogger(__name__)


class Task(abc.ABC):
    def __init__(
            self,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        self.device_env = device_env
        self.monitor = monitor

    @abc.abstractmethod
    def step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def end(self):
        pass


class TaskEval(Task):  
    def __init__(
            self,
            cfg: TaskEvalCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(device_env=device_env, monitor=monitor)
    
    def collate_fn(self, batch):
        pass

    def setup(self, *args, **kwargs):
        pass

    def prepare_for_evaluation(self):
        pass

    def end(self):
        pass


class TaskTrain(Task):
    def __init__(
            self,
            cfg: TaskTrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        self.num_intervals = cfg.num_intervals
        self.num_warmup_intervals = cfg.num_warmup_intervals
        self.log_frequency = cfg.log_frequency
        self.metrics_frequency = cfg.metrics_frequency
        self.num_steps_per_interval = None  # uninitialized, needs dataset info

        self.step_idx = 0  # current step (aka optimizer, scheduler update)
        self.batch_idx = 0  # current train batch (step % accum steps)
        self.interval_idx = 0  # interval (aka epoch or restorable period-between-checkpoints)
        self.interval_batch_idx = 0  # batch within current interval

        self.model = None
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type,
        #  we may want to differentiate different precision modes such as
        #  amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.amp:
            if cfg.dtype is not None and cfg.dtype in ('bfloat16', 'bf16'):
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        self.has_no_sync = False

        # optimization state initialized in setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.autocast = None

    def collate_fn(self, batch):
        pass

    def setup(
            self,
            num_batches_per_interval: int,
            resume_path: Optional[str] = None,
    ):
        """
        FIXME this interface needs refinement
        * currently, training duration is 'interval' based, where interval is either full dataset epoch, or
            sampled with replacement periods, intervals correspond to checkpoint / eval periods
        * LR schedule is updated per-step, so num_steps_per_interval is required to translate intervals ->
            total steps for scheduling
        * future should allow for step based durations (keeping interval as option), where train and warmup
            durations are specified in steps, checkpoint intervals in steps or time

        Args:
            num_batches_per_interval: pass info from data pipeline for LR scheduling
            resume_path: path for train state resume checkpoint
        Returns:

        """
        self._setup_model()

        if self.cfg.dist.dist_mode == 'fsdp':
            # FSDP makes the setup more complicated than DDP, need to shard first
            # Wrapper based changes such as .module, _fsdp_wrapped_module. and param name
            # changes need to be considered?
            # TODO Checkpoint saving needs FSDP support
            if self.device_env.world_size > 1:
                self._distribute_model()

            self._setup_optimization(
                num_batches_per_interval=num_batches_per_interval,
            )

            # TODO support resume
            assert not resume_path
        else:
            # DDP or no parallelization
            self._setup_optimization(
                num_batches_per_interval=num_batches_per_interval,
            )

            if resume_path:
                state_dict = load_checkpoint(resume_path)
                self.load_state_dict(state_dict)

            if self.device_env.world_size > 1:
                self._distribute_model()

    def _setup_model(self):
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense
        # in setup here vs in __init__()
        self.model.to(self.device_env.device)

        if self.cfg.opt.grad_checkpointing:
            self.model.set_grad_checkpointing()

    def _distribute_model(self):
        # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend
        # model size capacity beyond DDP w/o overloading HF cluster NCCL throughput.
        if self.cfg.dist.dist_mode == 'fsdp':
            self.model = fsdp_wrap_model(
                self.model,
                layers=self.model.get_wrap_layers(),
                device=self.device_env.device,
                sharding_strategy=self.cfg.dist.sharding_strategy,
                mixed_precision_dtype=self.amp_dtype,
            )
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device_env.device],
                static_graph=True,
            )
            self.has_no_sync = hasattr(self.model, 'no_sync')

    def _setup_optimization(self, num_batches_per_interval):
        device = self.device_env.device
        opt_kwargs = {}
        if self.cfg.opt.betas is not None:
            opt_kwargs["betas"] = self.cfg.opt.betas
        if self.cfg.opt.momentum is not None:
            opt_kwargs["momentum"] = self.cfg.opt.momentum
        self.optimizer = create_optimizer_v2(
            self.model,
            self.cfg.opt.optimizer,
            lr=self.cfg.opt.learning_rate,
            eps=self.cfg.opt.eps,
            layer_decay=self.cfg.opt.layer_decay,
            weight_decay=self.cfg.opt.weight_decay,
            **opt_kwargs,
        )

        if self.cfg.amp:
            self.scaler = timm.utils.NativeScaler()
            self.autocast = partial(
                torch.autocast, device_type=device.type, dtype=self.amp_dtype
            )
        else:
            self.scaler = None
            self.autocast = nullcontext

        # FIXME will need two paths here to support interval vs step based durations
        #  in either case LR is always stepped with each optimizer update (train step)
        self.num_steps_per_interval = num_batches_per_interval // self.cfg.opt.grad_accum_steps
        self.scheduler, num_scheduled_epochs = create_scheduler_v2(
            self.optimizer,
            self.cfg.opt.scheduler,
            warmup_lr=self.cfg.opt.warmup_learning_rate,
            warmup_epochs=self.num_warmup_intervals,
            num_epochs=self.num_intervals,
            step_on_epochs=False,  # sched is stepped on updates
            updates_per_epoch=self.num_steps_per_interval,
        )
        self.scheduler.step_update(0)

    def interval_start(self):
        # epoch / interval start hook, useful?
        self.model.train()
        self.optimizer.zero_grad()
        self.interval_batch_idx = 0

    def interval_end(self):
        # epoch / interval end hook, useful?
        self.monitor.log_phase('train', self.interval_idx)
        self.interval_idx += 1

    def step(self, sample: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict[str, Any]], ...]:
        output, loss, updated = self._update(sample)
        self._after_update(updated)
        return output, loss

    def _update(self, sample: Dict[str, Any]):
        accum_steps = self.cfg.opt.grad_accum_steps
        need_update = (self.interval_batch_idx + 1) % accum_steps == 0
        if self.has_no_sync and not need_update:
            with self.model.no_sync():
                output, loss = self._forward(sample)
                train_loss = self._loss(loss)
                if accum_steps > 1:
                    train_loss /= accum_steps
                self._backward(train_loss, need_update=False)
        else:
            output, loss = self._forward(sample)
            train_loss = self._loss(loss)
            if accum_steps > 1:
                train_loss /= accum_steps
            self._backward(train_loss)
        return output, loss, need_update

    def _after_update(self, updated=True):
        self.batch_idx += 1
        self.interval_batch_idx += 1
        if not updated:
            return
        self.step_idx += 1
        self.scheduler.step_update(self.step_idx)
        self.optimizer.zero_grad()

    @abc.abstractmethod
    def _forward(self, sample: Dict[str, Any]):
        """ model + loss forward pass
        """

    def _backward(self, loss, need_update=True):
        """ backward pass w/ gradient scaling & clipping
        """
        if self.scaler is not None:
            self.scaler(
                loss,
                self.optimizer,
                clip_grad=self.cfg.opt.clip_grad_value,
                clip_mode=self.cfg.opt.clip_grad_mode,
                parameters=self.model.parameters(),
                need_update=need_update,
            )
        else:
            loss.backward()
            if need_update:
                if self.cfg.opt.clip_grad_value is not None:
                    timm.utils.dispatch_clip_grad(
                        self.model.parameters(),
                        value=self.cfg.opt.clip_grad_value,
                        mode=self.cfg.opt.clip_grad_mode,
                    )
                self.optimizer.step()

    def _loss(self, loss):
        # extract the combined train loss for backward
        if isinstance(loss, (tuple, list)):
            return loss[0]
        elif isinstance(loss, dict):
            return loss['loss']
        return loss

    def _log_step(self, output, loss):
        self.monitor.log_step(
            'train',
            step_idx=self.step_idx,
            step_end_idx=self.num_intervals * self.num_steps_per_interval,
            interval=self.interval_idx,
            loss=self._loss(loss).item(),
            lr=self.get_current_lr(),
        )

    def after_step(self, sample, output, loss):
        """ operations to perform after step
        Anything that
          * needs to move tensor -> cpu (ie .item())
          * needs to execute code that is not torch.compile() safe
          * perform logging / io, etc
        """
        if self.step_idx % self.log_frequency == 0:
            self._log_step(output, loss)

    def get_current_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return lr

    def state_dict(self):
        state_dicts = {}
        state_dicts['model'] = self.model.state_dict()
        state_dicts['optimizer'] = self.optimizer.state_dict()
        if hasattr(self.scheduler, 'state_dict'):
            state_dicts['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state_dicts['scaler'] = self.scaler.state_dict()
        state_dicts['step_idx'] = self.step_idx
        state_dicts['batch_idx'] = self.batch_idx
        state_dicts['interval_idx'] = self.interval_idx
        return state_dicts

    def load_state_dict(
            self,
            state_dict,
            start_interval=None,
            restore_optimizer_state=True,
    ):
        def _clean(_sd):
            return {k[7:] if k.startswith('module.') else k: v for k, v in _sd.items()}

        # assume state_dict contains only model if key not present
        state_dict_model = state_dict.get('model', state_dict)
        if not hasattr(self.model, 'module'):
            state_dict_model = _clean(state_dict_model)
        self.model.load_state_dict(state_dict_model)

        if 'optimizer' not in state_dict:
            return

        # restore optimization related state
        if restore_optimizer_state:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            if 'scaler' in state_dict:
                self.scaler.load_state_dict(state_dict['scaler'])

        # restore LR schedule / step / interval related state
        if 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if 'step_idx' in state_dict:
            self.step_idx = state_dict['step_idx']
        if 'batch_idx' in state_dict:
            self.batch_idx = state_dict['batch_idx']
        if start_interval is not None:
            # override saved interval
            self.interval_idx = start_interval
        elif 'interval_idx' in state_dict:
            self.interval_idx = state_dict['interval_idx']

from dataclasses import dataclass
from functools import partial
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from transformers import AutoTokenizer

import timm
import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from . import OptimizationCfg
from .task import Task

from pixparse.framework.device import DeviceEnv
from pixparse.models import Cruller, ModelCfg
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno


# FIXME structure of config tree
# pull together model + prec + opt in a Task config that is then in the train cfg?
# or flatten model/prec/opt to train?


@dataclass
class TaskCrullerPretrainConfig:
    model: ModelCfg = ModelCfg()
    # tokenizer = ?  # FIXME tokenizer config needed?
    opt: OptimizationCfg = OptimizationCfg()
    dtype: Optional[str] = None
    amp: bool = True


class TaskCrullerPretrain(Task):
    def __init__(
            self,
            cfg: TaskCrullerPretrainConfig,
            device_env: DeviceEnv,
    ):
        super().__init__()
        self.cfg = cfg
        self.device_env = device_env
        self.task_start_token = '<s_pretrain>'
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length

        self.model = Cruller(cfg.model)
        self.model.to(device_env.device)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        # FIXME setup DDP / FSDP
        self.has_no_sync = False

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.Normalize(
                mean=(0.5,) * 3,
                std=(0.5,) * 3
            )
        ])
        self.image_preprocess_eval = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text_decoder.name)
        # Setup task specific tokens
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating
        # through full dataset on train start due to not being able to update once tokenizers
        # passed through to dataloader processes, we should store this all in configs up front
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))
        self.vocab_size = len(self.tokenizer)

        #self.anno_preprocess_train = preprocess_ocr_anno()
        self.anno_preprocess_train = partial(
            preprocess_text_anno,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.anno_preprocess_eval = None

        # optimization state initialized in train_setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.autocast = None

        # FIXME train/eval metrics and meters
        self.train_metrics = {}
        self.eval_metrics = {}

        # FIXME improve naming 'count', 'index'?
        self.step = 0  # step (aka update) count
        self.interval = 0  # interval (aka epoch or restorable period-between-checkpoints)
        self.interval_batch = 0  # batch count in current interval

    def train_setup(
            self,
            num_intervals: int,
            num_warmup_intervals: int,
            num_steps_per_interval: int,
    ):
        self.optimizer = create_optimizer_v2(
            self.model,
            self.cfg.opt.optimizer,
            lr=self.cfg.opt.learning_rate,
        )

        self.scheduler, num_scheduled_epochs = create_scheduler_v2(
            self.optimizer,
            self.cfg.opt.scheduler,
            step_on_epochs=False,
            updates_per_epoch=num_steps_per_interval,
            warmup_epochs=num_warmup_intervals,
            num_epochs=num_intervals,
        )
        self.scheduler.step_update(0)

        if self.cfg.amp:
            self.scaler = timm.utils.NativeScaler()
            self.autocast = partial(torch.autocast, device_type='cuda', dtype=self.cfg.dtype)
        else:
            self.scaler = None
            self.autocast = nullcontext

    def train_interval_start(self):
        # epoch / interval start hook, useful?
        self.optimizer.zero_grad()
        self.interval_batch = 0

    def train_interval_end(self):
        # epoch / interval end hook, useful?
        self.interval += 1

    def train_step(self, sample):
        image_input, text_input, text_target = sample
        result = {}

        image_input = image_input.to(self.device_env.device)
        text_input = text_input[:, :-1].to(self.device_env.device)
        text_target = text_target[:, 1:].to(self.device_env.device)

        accum_steps = self.cfg.opt.grad_accum_steps
        need_update = (self.interval_batch + 1) % accum_steps == 0

        def _forward():
            with self.autocast():
                output = self.model(image_input, text_input)
                logits = output['logits']
                loss = self.loss(
                    logits.view(-1, self.vocab_size),
                    text_target.view(-1),
                )
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if self.scaler is not None:
                self.scaler(
                    _loss,
                    self.optimizer,
                    clip_grad=self.cfg.opt.clip_grad_value,
                    clip_mode=self.cfg.opt.clip_grad_mode,
                    parameters=self.model.parameters(),
                    need_update=need_update,
                )
            else:
                _loss.backward()
                if need_update:
                    if self.cfg.opt.clip_grad_value is not None:
                        timm.utils.dispatch_clip_grad(
                            self.model.parameters(),
                            value=self.cfg.opt.clip_grad_value,
                            mode=self.cfg.opt.clip_grad_mode,
                        )
                    self.optimizer.step()

        if self.has_no_sync and not need_update:
            with self.model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        self.interval_batch += 1
        if not need_update:
            return result

        self.step += 1
        self.scheduler.step_update(self.step)
        self.optimizer.zero_grad()

        if self.step % 100 == 0:
            print(self.step, loss.item())

        return result

    def eval_step(self, sample):
        pass

    def state_dict(self):
        sd = {}
        sd['model'] = self.model.state_dict()
        sd['optimizer'] = self.optimizer.state_dict()
        if hasattr(self.scheduler, 'state_dict'):
            sd['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            sd['scaler'] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        outputs = [
            f'model: {repr(self.model)}',
            f'opt: {repr(self.optimizer)}',
            f'sched: {repr(self.scheduler)}',
        ]
        return '\n'.join(outputs)

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Optional, List, Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import timm
import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from pixparse.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerHF, TokenizerCfg
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from timm.layers import SelectAdaptivePool2d

from typing import Dict, List

from collections import OrderedDict

from ast import literal_eval 

_logger = logging.getLogger(__name__)


class GetCLSToken(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


@dataclass
class TaskCrullerFinetuneCORDCfg(TaskTrainCfg):
    model_name: Optional[
        str
    ] = None  # if model_name set, loads a pre-defined config in models/configs
    model: ModelCfg = field(
        default_factory=ModelCfg
    )  # FIXME rename model_cfg to diff from model_name?
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)

    def __post_init__(self):
        # FIXME figure out how to get command line args to overlay on top pre-defined
        # config but ONLY if they are specified on cmd line?
        if self.model_name:
            model = get_model_config(self.model_name)
            if model is None:
                _logger.warning(
                    f"Model config for {self.model_name} was not found, using defaults."
                )
            else:
                self.model = model
        else:
            self.model_name = "custom"


class TaskCrullerFinetuneCORD(TaskTrain):
    def __init__(
        self,
        cfg: TaskCrullerFinetuneCORDCfg,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type,
        #  we may want to differentiate different precision modes such as
        #  amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (
                torch.bfloat16 if cfg.dtype in ("bfloat16", "bf16") else torch.float16
            )

        self.task_start_token = "<s_cord>"
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = TokenizerHF(cfg.tokenizer)

        self.state_dict = OrderedDict()
        self.resume = False

        # Setup task specific tokens
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating
        # through full dataset on train start due to not being able to update once tokenizers
        # passed through to dataloader processes, we should store this all in configs up front
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )


        self.vocab_size = len(self.tokenizer.trunk)

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer.trunk,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model)  # FIXME would be good to defer weight init here

        # We need to resize the token embeddings after the model has been initialized
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

        # TODO refactor, used in many tasks

        img_mean = self.model.image_encoder.trunk.pretrained_cfg["mean"]
        img_std = self.model.image_encoder.trunk.pretrained_cfg["std"]

        self.img_mean = (
            sum(img_mean) / len(img_mean)
            if cfg.model.image_encoder.image_fmt == "L"
            else img_mean
        )
        self.img_std = (
            sum(img_std) / len(img_std)
            if cfg.model.image_encoder.image_fmt == "L"
            else img_std
        )

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    cfg.model.image_encoder.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.CenterCrop(448),  # FIXME need better aspect preserving resize & pad
                transforms.Normalize(
                    mean=self.img_mean,
                    std=self.img_std,
                ),
            ]
        )

    def train_setup(
        self,
        num_batches_per_interval: int,
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
            num_batches_per_interval:

        Returns:

        """
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense
        # in setup here vs in __init__(). For __init__ need the model structure to
        # instantiate / setup tokenizer, other aspects. I don't think we need to init
        # weights / move to device until here.
        device = self.device_env.device
        self.model.to(device)

        if self.device_env.world_size > 1:
            # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend
            # model size capacity beyond DDP w/o overloading HF cluster NCCL throughput.
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device],
                static_graph=True,
            )
            self.has_no_sync = hasattr(self.model, "no_sync")

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
        self.num_steps_per_interval = (
            num_batches_per_interval // self.cfg.opt.grad_accum_steps
        )
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

    def json2token(self, obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence.
        """
        # FIXME Currently broken, was outputting broken apart pieces of json hierarchies

        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.tokenizer.trunk.add_special_tokens({"additional_special_tokens":[fr"<s_{k}>", fr"</s_{k}>"]})
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.tokenizer.trunk.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def collate_fn(self, batch):
        """
        basic collator for PIL images, as returned by rvlcdip dataloader (among others)
        """
        images = [item["image"] for item in batch]
        text_inputs = [literal_eval(item["ground_truth"])['gt_parse'] for item in batch]
        text_inputs = [self.task_start_token + self.json2token(text) for text in text_inputs]
        transform = self.image_preprocess_train
        images = torch.stack([transform(img) for img in images])
        return {"image": images, "label": text_inputs, "text_target": text_inputs} #FIXME need to remove pad tokens + irrelevant computational tokens

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_input = sample["image"]
        label = sample["label"]
        text_target = sample["text_target"]
        result = {}

        image_input = image_input.to(self.device_env.device, non_blocking=True)
        label = label.to(self.device_env.device, non_blocking=True)
        text_target = text_target.to(self.device_env.device, non_blocking=True)
        
        accum_steps = self.cfg.opt.grad_accum_steps
        need_update = (self.interval_batch_idx + 1) % accum_steps == 0

        def _forward():
            with self.autocast():
                output = self.model(image_input, label)
                logits = output["logits"]
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

        self.batch_idx += 1
        self.interval_batch_idx += 1
        if self.step % self.eval_frequency == 0:
            self.monitor.log_step(
                "finetune",
                step_idx=self.step,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=None,
                eval_data=None,
            )

        if not need_update:
            return result

        self.step += 1
        self.scheduler.step_update(self.step)
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        state_dicts["tokenizer"] = self.tokenizer.state_dicts() # Needed because of dynamic updates for the tokenizer
        return state_dicts
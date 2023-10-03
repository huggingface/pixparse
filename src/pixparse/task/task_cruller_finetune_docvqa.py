import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import timm
import timm.utils
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pixparse.data.loader import BaseCollate
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno, text_input_to_target
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, TaskTrainCfg
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerCfg, create_tokenizer

_logger = logging.getLogger(__name__)


class CollateDocVQA(BaseCollate):
    """
    basic collator for PIL images, as returned by docVQA dataloader (among others)
    """

    def __init__(
        self,
        tokenizer,
        image_preprocess,
        start_token,
        max_length: int,
        end_token,
    ):
        super().__init__(
            tokenizer, image_preprocess, start_token, max_length=max_length
        )
        self.end_token = end_token

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        # question/answer tokens are already present in the data
        q_and_as = [np.random.choice(item['labels']) for item in batch]  # TODO allow to change strategy here
        labels_tokens = []
        for text in q_and_as:
            labels_tokens.append(self.tokenizer_fn(
                self.start_token
                + text
                + self.tokenizer.eos_token
            ))
        return self.pack_inputs(images, labels_tokens)

    def pack_inputs(self, images, labels_tokens):
        images = torch.stack([self.image_preprocess(img) for img in images])
        labels = torch.stack(labels_tokens)
        targets = torch.stack(
            [
                text_input_to_target(
                    text_input=text,
                    tokenizer=self.tokenizer,
                    prompt_end_token=self.end_token,
                )
                for text in labels
            ]
        )
        labels = labels[:, :-1]
        targets = targets[:, 1:]

        return {"image": images, "label": labels, "text_target": targets}


@dataclass
class TaskCrullerFinetuneDOCVQACfg(TaskTrainCfg):
    model_name: Optional[
        str
    ] = None  # if model_name set, loads a pre-defined config in models/configs
    model: ModelCfg = field(
        default_factory=ModelCfg
    )  # FIXME rename model_cfg to diff from model_name?
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)

    def __post_init__(self):
        # FIXME figure out how to get command line args to overlay on top pre-defined config but ONLY if they are
        # specified on cmd line?
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


class TaskCrullerFinetuneDOCVQA(TaskTrain):
    def __init__(
        self,
        cfg: TaskCrullerFinetuneDOCVQACfg,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type, we may want to
        #  differentiate different precision modes such as amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (
                torch.bfloat16 if cfg.dtype in ("bfloat16", "bf16") else torch.float16
            )

        self.task_start_token = "<s_docvqa>"
        self.prompt_end_token = "<s_answer>"  # Slice prompt right before answer content
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = create_tokenizer(cfg.tokenizer)

        self.state_dict = OrderedDict()
        self.resume = False

        # Setup task specific tokens NOTE: Donut appears to add tokens on the fly during dataset init, requires
        # iterating through full dataset on train start due to not being able to update once tokenizers passed through
        # to dataloader processes, we should store this all in configs up front
        self.special_tokens_finetune = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
            "<s_question>",
            "</s_question>",
            "</s_answer>",
        ]

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model)  # FIXME would be good to defer weight init here

        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]

        num_tokens_from_pretrain = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )
        # need to resize embeddings from pretrained model in order to load it
        if num_tokens_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer)
            )

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

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

        # preprocessors cross both the task/model & dataset domain, created within task here and passed to data loaders

        image_size = cfg.model.image_encoder.image_size
        color_transform = transforms.Grayscale()

        self.image_preprocess_train = transforms.Compose(
            [
                transforms.ToTensor(),
                color_transform,
                transforms.Resize(
                    image_size,
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

    def setup(
        self,
        num_batches_per_interval: int,
    ):
        """
        FIXME this interface needs refinement * currently, training duration is 'interval' based, where interval is
        either full dataset epoch, or
            sampled with replacement periods, intervals correspond to checkpoint / eval periods
        * LR schedule is updated per-step, so num_steps_per_interval is required to translate intervals ->
            total steps for scheduling
        * future should allow for step based durations (keeping interval as option), where train and warmup
            durations are specified in steps, checkpoint intervals in steps or time

        Args:
            num_batches_per_interval:

        Returns:

        """
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense in setup here vs in __init__(). For
        # __init__ need the model structure to instantiate / setup tokenizer, other aspects. I don't think we need to
        # init weights / move to device until here if self.resume:
        _logger.info("Resuming from existing checkpoint.")
        self.state_dict = {k.replace("module.", ""): v for k, v in self.state_dict.items()}
        self.model.load_state_dict(self.state_dict)
        self.newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(self.special_tokens_finetune))}
        )
        self.vocab_size = len(self.tokenizer)

        # We resize token embeddings after initializing
        if self.newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer)
            )

        device = self.device_env.device
        self.model.to(device)

        if self.device_env.world_size > 1:
            # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend model size capacity
            #  beyond DDP w/o overloading HF cluster NCCL throughput.
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device],
                static_graph=True,
            )
            self.has_no_sync = hasattr(self.model, "no_sync")

        self._setup_optimization(num_batches_per_interval)

    def collate_fn(self, batch):
        return CollateDocVQA(
            self.tokenizer,
            self.image_preprocess_train,
            self.task_start_token,
            end_token=self.prompt_end_token
        )

    def step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
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
                    self.optimizer.step_idx()

        if self.has_no_sync and not need_update:
            with self.model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)
        self.batch_idx += 1
        self.interval_batch_idx += 1
        if self.step % 100 == 0:
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
        state_dicts[
            "tokenizer"
        ] = (
            self.tokenizer.state_dict()
        )  # FIXME not needed anymore? we preprocess everything before
        return state_dicts

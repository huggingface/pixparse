import logging
from ast import literal_eval
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import timm
import timm.utils
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno, create_transforms
from pixparse.data.loader import BaseCollate
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, TaskTrainCfg
from pixparse.models import Cruller, create_model, ModelArgs
from pixparse.tokenizers import TokenizerCfg, create_tokenizer
from pixparse.utils.json_utils import json2token, token2json, JSONParseEvaluator  # assuming you need all three


_logger = logging.getLogger(__name__)


class CollateDocVQA(BaseCollate):
    r"""
    A collator for handling batches of PIL images and corresponding labels
    as utilized with the SinglePageDocVQA dataset.
    Strings returned will be <s_docvqa><s><question>...question...?</s_question><s_answer>...answer.<s_answer></s>
    Args:
        tokenizer (`Callable`):
            Tokenizer function to convert textual labels into tokens.
        image_preprocess (`Callable`):
            method to perform preprocessing operations on images.
        start_token (`str`):
            A token that indicates the start of a sequence from the current task. <s_rvlcdip> for RVLCDIP, etc.
        max_length (`int`):
            Maximum length allowed for tokenized text sequences.
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
            labels_tokens.append(self.tokenizer_fn(self.start_token + text + self.tokenizer.eos_token))
        return self.pack_inputs(images, labels_tokens)


@dataclass
class TaskCrullerFinetuneDOCVQACfg(TaskTrainCfg):
    model: ModelArgs = field(default_factory=lambda: ModelArgs(
        name='cruller_base',  # override model default in spec per task
    ))

    def __post_init__(self):
        assert self.model.cfg is not None
        if self.tokenizer is None:
            # set tokenizer to text tower model name if not explicitly set
            self.tokenizer = TokenizerCfg(name=self.model.cfg.text_decoder.name)


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
        model_cfg = self.cfg.model.cfg
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type, we may want to
        #  differentiate different precision modes such as amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (torch.bfloat16 if cfg.dtype in ("bfloat16", "bf16") else torch.float16)

        self.task_start_token = "<s_docvqa>"
        self.prompt_end_token = "<s_answer>"  # Slice prompt right before answer content
        self.max_position_embeddings = model_cfg.text_decoder.max_length
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
        self.model = create_model(
            model_cfg,
            pretrained='',  # FIXME pass through tags or paths for full pretrained image-text tower
        )

        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]

        num_tokens_from_pretrain = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )
        # need to resize embeddings from pretrained model in order to load it
        if num_tokens_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        # preprocessors cross both the task/model & dataset domain, created within task here and passed to data loaders

        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.image_preprocess_train = create_transforms(
            self.cfg.image_transforms,
            input_cfg=self.image_input_cfg,
            training=True,
            interpolation='bicubic',
            crop_margin=False,  # True?
            align_long_axis=False,
        )

    def setup(self, num_batches_per_interval: int, ):
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
        self.collator = CollateDocVQA(
            self.tokenizer,
            self.image_preprocess_train,
            self.task_start_token,
            end_token=self.prompt_end_token,
            max_length=512  # FIXME derive from config
        )
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense in setup here vs in __init__(). For
        # __init__ need the model structure to instantiate / setup tokenizer, other aspects. I don't think we need to
        # init weights / move to device until here if self.resume:

        _logger.info("Resuming from existing checkpoint.")

        checkpoint = torch.load(
            "/fsx/pablo/training_pixparse/20231023-094042-task_cruller_pretrain-model_cruller_swin_384_to_1920-lr_3.0e-05-b_16/checkpoint-29.pt")
        self.state_dict = checkpoint["model"]
        self.state_dict = {k.replace("module.", ""): v for k, v in self.state_dict.items()}
        self.model.load_state_dict(self.state_dict)
        device = self.device_env.device
        self.model.to(device)

        # FIXME derive from config "finetune checkpoint"

        self.newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(self.special_tokens_finetune))}
        )
        self.vocab_size = len(self.tokenizer)

        # We resize token embeddings after initializing
        if self.newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))

        if self.device_env.world_size > 1:
            # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend model size capacity
            #  beyond DDP w/o overloading HF cluster NCCL throughput.
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device], static_graph=True)
            self.has_no_sync = hasattr(self.model, "no_sync")

        self._setup_optimization(num_batches_per_interval)

    def collate_fn(self, batch):
        return self.collator(batch)

    def _forward(self, sample: Dict[str, Any]):
        image_input = sample["image"]
        label = sample["label"]
        text_target = sample["text_target"]
        image_input = image_input.to(self.device_env.device, non_blocking=True)
        label = label.to(self.device_env.device, non_blocking=True)
        text_target = text_target.to(self.device_env.device, non_blocking=True)

        with self.autocast():
            output = self.model(image_input, label)
            logits = output["logits"]
            loss = self.loss(
                logits.view(-1, self.vocab_size),
                text_target.view(-1),
            )
        return output, loss

    def after_step(self, sample, output, loss):
        metrics_updated = False
        if self.step_idx % self.metrics_frequency == 0:
            # TODO add metrics and possibly eval_gallery for finetuning
            self.train_metrics = None
            eval_gallery = None
            metrics_updated = True

        if metrics_updated or self.step_idx % self.log_frequency == 0:
            self.monitor.log_step(
                'train',
                step_idx=self.step_idx,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=self.train_metrics if metrics_updated else None,
                eval_data=eval_gallery if metrics_updated else None,
            )

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        state_dicts["tokenizer"] = (self.tokenizer.state_dict())
        # FIXME not needed anymore? we preprocess everything before
        return state_dicts

    def __repr__(self):
        outputs = [
            f'model: {repr(self.model)}',
            f'opt: {repr(self.optimizer)}',
            f'sched: {repr(self.scheduler)}',
        ]
        return '\n'.join(outputs)

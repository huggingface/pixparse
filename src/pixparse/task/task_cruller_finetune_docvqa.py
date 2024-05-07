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
from pixparse.models import Cruller, create_model, resize_model_embeddings, ModelArgs
from pixparse.tokenizers import TokenizerCfg, create_tokenizer
from pixparse.utils.json_utils import json2token, token2json, JSONParseEvaluator  # assuming you need all three
from pixparse.utils import load_checkpoint, get_latest_checkpoint
from PIL import Image
import io
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
        #images = [Image.open(io.BytesIO(item["image"]['bytes'])) for item in batch]
        images = [item["image"] for item in batch]
        # question/answer tokens are already present in the data
        
        questions = [item["question"] for item in batch] # list of questions
        answers = [np.random.choice(item["answers"]) for item in batch] # select one answer per question
        questions_and_answers = []
        for question, answer in zip(questions, answers):
            questions_and_answers.append("<s_question>" + question + "</s_question><s_answer>" + answer + "</s_answer>")

        labels_tokens = []
        for text in questions_and_answers:
            labels_tokens.append(self.tokenizer_fn(self.start_token + text + self.tokenizer.eos_token))
        return self.pack_inputs(images, labels_tokens)


@dataclass
class TaskCrullerFinetuneDOCVQACfg(TaskTrainCfg):
    # TODO do we want to pass here all parameters defined before?
    model: ModelArgs = field(default_factory=lambda: ModelArgs(
        name='cruller_base', text_max_length=1024  # override model default in spec per task
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
        checkpoint_path: str = "",
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

        # Setup task specific tokens NOTE: Donut appears to add tokens on the fly during dataset init, requires
        # iterating through full dataset on train start due to not being able to update once tokenizers passed through
        # to dataloader processes, we should store this all in configs up front
        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno

        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]
        
        num_pretrain_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )
        self.model = create_model(
            model_cfg,
            pretrained=checkpoint_path, 
            new_vocab_size=len(self.tokenizer)
        )
        finetuning_special_tokens = [
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
            "<s_question>",
            "</s_question>",
            "</s_answer>",
        ]

        num_finetuning_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(finetuning_special_tokens))}, replace_additional_special_tokens=False
        )
        if num_finetuning_tokens > 0:
            resize_model_embeddings(self.model, new_vocab_size=len(self.tokenizer))

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        # preprocessors cross both the task/model & dataset domain, created within task here and passed to data loaders

        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.image_preprocess_train = create_transforms(
            self.cfg.image_transforms,
            input_cfg=self.image_input_cfg,
            training=True,
            interpolation='bicubic',
            crop_margin=False,  # True?
            align_long_axis=True,
        )

        self.collator = CollateDocVQA(
            self.tokenizer,
            self.image_preprocess_train,
            start_token=self.task_start_token,
            end_token=self.prompt_end_token,
            max_length=128  # FIXME derive from config
        )

    def setup(self, num_batches_per_interval: int, resume: str=""):
        """
        Overrides the setup method to add additional setup steps specific to TaskCrullerFinetuneDOCVQA.
        The additional setup step is adding finetuning tokens.

        Args:
            num_batches_per_interval: Number of batches per interval.
            resume: Flag indicating whether to resume from a checkpoint, if present, its path or "latest".
        """

        if resume:
            if resume == "latest":
                resume = get_latest_checkpoint(resume)
            state_dict = load_checkpoint(resume)
            print(f"Currently resuming from {resume}")
            self.load_state_dict(state_dict)

        self.model.text_decoder.trunk.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id

        super().setup(num_batches_per_interval=num_batches_per_interval, resume=resume)

    def collate_fn(self, batch):
        return self.collator(batch)

    def _forward(self, sample: Dict[str, Any]):
        image_input = sample["image"]
        label = sample["label"]
        text_target = sample["text_target"]
        image_input = image_input.to(self.device_env.device, non_blocking=True)
        label = label.to(self.device_env.device, non_blocking=True)
        text_target = text_target.to(self.device_env.device, non_blocking=True)
        # forward method

        with self.autocast():
            output = self.model(image_input, label)
            logits = output["logits"]
            loss = self.loss(
                logits.view(-1, len(self.tokenizer)),
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

    def __repr__(self):
        outputs = [
            f'model: {repr(self.model)}',
            f'opt: {repr(self.optimizer)}',
            f'sched: {repr(self.scheduler)}',
        ]
        return '\n'.join(outputs)

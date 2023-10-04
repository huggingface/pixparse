import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict

import torch.nn as nn
import torchvision.transforms as transforms

from pixparse.data import (preprocess_ocr_anno, preprocess_text_anno)
from pixparse.data.loader import BaseCollate
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, TaskTrainCfg
from pixparse.models import Cruller, ModelArgs
from pixparse.tokenizers import TokenizerCfg, create_tokenizer

_logger = logging.getLogger(__name__)


class GetCLSToken(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class CollateRVLCDIP(BaseCollate):
    r"""
    A collator for handling batches of PIL images and corresponding labels, as utilized with the RVL-CDIP dataset.
    Converts class labels to tokens and returns a string.
    Args:
        tokenizer (`Callable`):
            Tokenizer function to convert textual labels into tokens.
        image_preprocess (`Callable`):
            method to perform preprocessing operations on images.
        start_token (`str`):
            A token that indicates the start of a sequence from the current task. <s_rvlcdip> for RVLCDIP, etc.
        max_length (`int`):
            Maximum length allowed for tokenized text sequences.
        label_int2str (`dict`):
            A mapping from integer labels to string representations.
    """

    def __init__(
        self,
        tokenizer,
        image_preprocess,
        start_token,
        max_length: int,
        label_int2str: dict,
    ):
        super().__init__(
            tokenizer, image_preprocess, start_token, max_length=max_length
        )
        self.int2str = label_int2str

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        labels_tokens = [
            self.tokenizer_fn(
                self.start_token
                + "<"
                + self.int2str[label]
                + "/>"
                + self.tokenizer.eos_token
            )
            for label in labels
        ]
        return self.pack_inputs(
            images,
            labels_tokens
        )


@dataclass
class TaskCrullerFinetuneRVLCDIPCfg(TaskTrainCfg):
    model: ModelArgs = field(
        default_factory=lambda: ModelArgs(
            name="cruller_base",  # override model default in spec per task
        )
    )

    def __post_init__(self):
        assert self.model.cfg is not None
        if self.tokenizer is None:
            # set tokenizer to text tower model name if not explicitly set
            self.tokenizer = TokenizerCfg(
                name=self.model.cfg.text_decoder.name)


class TaskCrullerFinetuneRVLCDIP(TaskTrain):
    def __init__(
        self,
        cfg: TaskCrullerFinetuneRVLCDIPCfg,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.task_start_token = "<s_rvlcdip>"
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = create_tokenizer(cfg.tokenizer)

        # Setup task specific tokens
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating through full dataset on
        # train start due to not being able to update once tokenizers passed through to dataloader processes,
        #  we should store this all in configs up front
        self.special_tokens_finetune = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
            "<s_class>",  # This and what follows is valid only for RVLCDIP task
            "</s_class>",
            "<advertisement/>",
            "<budget/>",
            "<email/>",
            "<file_folder/>",
            "<form/>",
            "<handwritten/>",
            "<invoice/>",
            "<letter/>",
            "<memo/>",
            "<news_article/>",
            "<presentation/>",
            "<questionnaire/>",
            "<resume/>",
            "<scientific_publication/>",
            "<scientific_report/>",
            "<specification/>",
        ]

        self.label_int2str = {
            0: "letter",
            1: "form",
            2: "email",
            3: "handwritten",
            4: "advertisement",
            5: "scientific_report",
            6: "scientific_publication",
            7: "specification",
            8: "file_folder",
            9: "news_article",
            10: "budget",
            11: "invoice",
            12: "presentation",
            13: "questionnaire",
            14: "resume",
            15: "memo",
        }

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        # FIXME would be good to defer weight init here
        self.model = Cruller(cfg.model)

        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]
        num_tokens_from_pretrain = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(
                set(special_tokens_from_pretrain))}
        )
        # need to resize embeddings from pretrained model in order to load it
        if num_tokens_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer))

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        # TODO refactor, used in many tasks
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

    def setup(
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
        _logger.info("Resuming from existing checkpoint.")
        self.state_dict = {
            k.replace("module.", ""): v for k, v in self.state_dict.items()
        }
        self.model.load_state_dict(self.state_dict)
        self.newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(
                set(self.special_tokens_finetune))}
        )
        self.vocab_size = len(self.tokenizer)

        # We resize token embeddings after initializing
        if self.newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer))

        self._setup_model()
        self._setup_optimization(
            num_batches_per_interval=num_batches_per_interval,
        )

    def collate_fn(self, batch):
        return CollateRVLCDIP(
            self.tokenizer,
            self.image_preprocess_train,
            self.task_start_token,
            self.label_int2str,
        )

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
        if self.step_idx % self.metrics_frequency == 0:
            self.monitor.log_step(
                "finetune",
                step_idx=self.step_idx,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=None,
                eval_data=None,
            )

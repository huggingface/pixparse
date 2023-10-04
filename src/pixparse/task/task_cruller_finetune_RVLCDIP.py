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

from pixparse.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor, OptimizationCfg
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import create_tokenizer, TokenizerCfg
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno, create_transforms
from timm.layers import SelectAdaptivePool2d

from typing import Dict, List

from collections import OrderedDict

_logger = logging.getLogger(__name__)


class GetCLSToken(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


def text_input_to_target(self, text_input, tokenizer, ignore_id=-100):
    target = text_input.clone()
    # model doesn't need to predict pad token
    target[target == tokenizer.pad_token_id] = ignore_id
    # model doesn't need to predict prompt (for VQA)
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(self.prompt_end_token)
    target[: torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id
    return target


class CollateRVLCDIP:
    """
    basic collator for PIL images, as returned by rvlcdip dataloader (among others)
    """

    def __init__(
            self,
            tokenizer,
            image_preprocess,
            start_token,
            label_int2str,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_fn = lambda x: self.tokenizer(
            x,
            add_special_tokens=False,
            return_tensors='pt',
            max_length=5,
            padding='max_length',
            truncation=True).input_ids[0]
        self.image_preprocess = image_preprocess
        self.start_token = start_token
        self.int2str = label_int2str

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        labels_tokens = [
            self.tokenizer_fn(self.start_token + "<" + self.int2str[label] + "/>" + self.tokenizer.eos_token)
            for label in labels
        ]
        images = torch.stack([self.image_preprocess(img) for img in images])
        labels = torch.stack(labels_tokens)
        targets = torch.stack([text_input_to_target(text, self.tokenizer) for text in labels])
        labels = labels[:, :-1]
        targets = targets[:, 1:]

        return {"image": images, "label": labels, "text_target": targets}


@dataclass
class TaskCrullerFinetuneRVLCDIPCfg(TaskTrainCfg):
    # if model_name set, loads a pre-defined config in models/configs
    model_name: Optional[str] = None
    model: ModelCfg = field(default_factory=ModelCfg)  # FIXME rename model_cfg to diff from model_name?
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    opt: OptimizationCfg = field(default_factory=lambda: OptimizationCfg(
        optimizer='nadamw',
        learning_rate=1e-4,
    ))

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
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating
        # through full dataset on train start due to not being able to update once tokenizers
        # passed through to dataloader processes, we should store this all in configs up front
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

        # TODO refactor, used in many tasks
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3
        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.image_preprocess_train = create_transforms(
            self.cfg.image_transforms,
            input_cfg=self.image_input_cfg,
            training=True,
            interpolation='bicubic',
            crop_margin=False,  # True?
            align_long_axis=False,
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
        _logger.info(f"Resuming from existing checkpoint. ")
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
            # TODO add metrics and possibly eval_gallery for finetuning
            self.train_metrics = None
            eval_gallery = None
            metrics_updated = True

        if self.step_idx % self.metrics_frequency == 0:
            self.monitor.log_step(
                "finetune",
                step_idx=self.step_idx,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=self.train_metrics if metrics_updated else None,
                eval_data=eval_gallery if metrics_updated else None,
            )

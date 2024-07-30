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
from pixparse.tokenizers import create_tokenizer, TokenizerCfg
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from timm.layers import SelectAdaptivePool2d

from typing import Dict, List

from collections import OrderedDict

_logger = logging.getLogger(__name__)


class GetCLSToken(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


@dataclass
class TaskCrullerFinetuneXentCfg(TaskTrainCfg):
    model_name: Optional[str] = None  # if model_name set, loads a pre-defined config in models/configs
    model: ModelCfg = field(default_factory=ModelCfg)  # FIXME rename model_cfg to diff from model_name?
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    def __post_init__(self):
        # FIXME figure out how to get command line args to overlay on top pre-defined
        # config but ONLY if they are specified on cmd line?
        if self.model_name:
            model = get_model_config(self.model_name)
            if model is None:
                _logger.warning(f'Model config for {self.model_name} was not found, using defaults.')
            else:
                self.model = model
        else:
            self.model_name = 'custom'


class TaskCrullerFinetuneXent(TaskTrain):
    def __init__(
            self,
            cfg: TaskCrullerFinetuneXentCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.task_start_token = '<s_finetune>'
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = False  # set for image-text dataset experiments
        self.tokenizer = create_tokenizer(cfg.tokenizer)

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
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )

        self.vocab_size = len(self.tokenizer)

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model)  # FIXME would be good to defer weight init here

        # We need to resize the token embeddings after the model has been initialized
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == 'L' else 3
        
        # TODO refactor, used in many tasks

        img_mean = self.model.image_encoder.trunk.pretrained_cfg['mean']
        img_std = self.model.image_encoder.trunk.pretrained_cfg['std']
        
        self.img_mean = sum(img_mean) / len(img_mean) if cfg.model.image_encoder.image_fmt == 'L' else img_mean
        self.img_std = sum(img_std) / len(img_std) if cfg.model.image_encoder.image_fmt == 'L' else img_std

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                cfg.model.image_encoder.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True),
            transforms.Normalize(
                mean=self.img_mean,
                std=self.img_std,
            )
        ])

        # "pool" results and setup classification head

    def setup(self, num_batches_per_interval: int):
        # Load model

        # First load base model, then specialize it to fine-tuning end
        
        # FIXME pass along resume arg here
        if self.resume:
            _logger.info(f"Resuming from existing checkpoint. ")
            self.state_dict = {k.replace("module.", ""): v for k, v in self.state_dict.items()}
            self.model.load_state_dict(self.state_dict)

        self.model = nn.Sequential(OrderedDict([
            ("encoder", self.model.image_encoder),
            ("token_pool", GetCLSToken()),
            ("final_fc", nn.Linear(768, 16)),  # 16 classes in RVLCDIP
            # nn.Softmax(16)
        ])
        )

        # weights / move to device until here.
        self._setup_model()
        self._setup_optimization(
            num_batches_per_interval=num_batches_per_interval,
        )

    def collate_fn(self, batch):
        """
        basic collator for PIL images, as returned by rvlcdip dataloader (among others)
        """
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        
        transform = self.image_preprocess_train
        
        images = torch.stack([transform(img) for img in images])
        labels = torch.tensor(labels, dtype=torch.int64)
        return {'image': images, 'label': labels}

    def interval_start(self):
        self.optimizer.zero_grad()
        self.interval_batch_idx = 0

    def interval_end(self):
        self.monitor.log_phase('finetune', self.interval_idx)
        self.interval_idx += 1

    def _forward(self, sample: Dict[str, Any]):
        image_input = sample['image']
        label = sample['label']

        image_input = image_input.to(self.device_env.device, non_blocking=True)
        label = label.to(self.device_env.device, non_blocking=True)
        with self.autocast():
            outputs = self.model(image_input)
            loss = self.loss(outputs, label)
        return outputs, loss

    def after_step(self, sample, output, loss):
        if self.step_idx % self.metrics_frequency == 0:
            self.monitor.log_step(
                'finetune',
                step_idx=self.step_idx,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=None,
                eval_data=None
            )

    def get_current_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return lr

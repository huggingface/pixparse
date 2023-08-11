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
from pixparse.utils.ocr_utils import get_ocr_metrics



_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerFinetuneCfg(TaskTrainCfg):
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

class TaskCrullerFinetune(TaskTrain):
    def __init__(
            self,
            cfg: TaskCrullerFinetuneCfg,
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
            self.amp_dtype = torch.bfloat16 if cfg.dtype in ('bfloat16', 'bf16') else torch.float16

        self.task_start_token = '<s_finetune>'
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = False  # set for image-text dataset experiments
        self.tokenizer = TokenizerHF(cfg.tokenizer)
        
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
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer.trunk))
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
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
            #transforms.CenterCrop(448),  # FIXME need better aspect preserving resize & pad
            transforms.Normalize(
                mean=self.img_mean,
                std=self.img_std,
            )
        ])

    def train_setup(self, *args, **kwargs):
        # weights / move to device until here.
        device = self.device_env.device
        self.model.to(device)

    def train_interval_start(self):
        self.optimizer.zero_grad()
        self.interval_batch_idx = 0

    def train_interval_end(self):
        pass

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def eval_step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # TODO Remove eval method from train dataclass
        pass

    def get_current_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return lr

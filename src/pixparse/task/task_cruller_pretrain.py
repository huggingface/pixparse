import logging
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pixparse.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor
from pixparse.models import create_model, ModelArgs
from pixparse.tokenizers import create_tokenizer, TokenizerCfg
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno, create_transforms
from pixparse.utils.ocr_utils import get_ocr_metrics


_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerPretrainCfg(TaskTrainCfg):
    model: ModelArgs = field(default_factory=lambda: ModelArgs(
        name='cruller_base',  # override model default in spec per task
    ))

    def __post_init__(self):
        assert self.model.cfg is not None
        if self.tokenizer is None:
            # set tokenizer to text tower model name if not explicitly set
            self.tokenizer = TokenizerCfg(name=self.model.cfg.text_decoder.name)


class TaskCrullerPretrain(TaskTrain):
    """ Cruller Pretraining Task

    NOTES:
      * all task code is currently here w/ nothing in base class but interface
      * we will want to pull out bits that are common to other tasks as we proceed
         by pushing into base classe(s), stand-alone fn / helper classes, etc.
      * to setup schedule we need info from data-pipeline re samples, etc so our call sequence is:
        * Task() -- task __init__() called for instance, setup what we can
        * Initialize data-pipeline (external to Task) to get batch / step count
        * Call setup() to pass this info back to Task and finish setting up optimizer / scheduler
        * Proceed to train by interval_start()/step() * N/interval_end(), etc
    """
    def __init__(
            self,
            cfg: TaskCrullerPretrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        model_cfg = self.cfg.model.cfg
        self.task_start_token = '<s_pretrain>'
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = model_cfg.text_decoder.max_length
        self.text_anno_fn = False  # set for image-text dataset experiments
        self.tokenizer = create_tokenizer(cfg.tokenizer)
        self.model = create_model(
            model_cfg,
            pretrained='',  # FIXME pass through tags or paths for full pretrained image-text tower
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Setup task specific tokens
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating
        # through full dataset on train start due to not being able to update once tokenizers
        # passed through to dataloader processes, we should store this all in configs up front
        # FIXME this token setup remains a bit of a mess, need better,
        #  re-usable scheme for updating pretrain vs fine-tune tokens in correct order
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )
        self.vocab_size = len(self.tokenizer)
        # We need to resize the token embeddings after the model has been initialized
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))

        # Setup text preprocessing
        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        # Setup image preprocessing
        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.image_preprocess_train = create_transforms(
            self.cfg.image_transforms,
            input_cfg=self.image_input_cfg,
            training=True,
            interpolation='bicubic',
            crop_margin=False,  # True?
            align_long_axis=False,
        )
        # TODO init eval when eval tasks are being created from train task
        self.image_preprocess_eval = None

        # TODO These metrics have to be organized as dicts of dicts. 
        # First level is the category, second level is the tag
        # We have to make this clear
        self.train_metrics = {} 
        self.max_recursion_length = 1000  # specific to Cruller for generation

    def _forward(self, sample: Dict[str, Any]):
        image_input, text_input, text_target = sample
        image_input = image_input.to(self.device_env.device, non_blocking=True)
        text_input = text_input[:, :-1].to(self.device_env.device, non_blocking=True)
        text_target = text_target[:, 1:].to(self.device_env.device, non_blocking=True)

        with self.autocast():
            output = self.model(image_input, text_input)
            logits = output['logits']
            loss = self.loss(
                logits.view(-1, self.vocab_size),
                text_target.view(-1),
            )
        return output, loss

    def after_step(self, sample, output, loss):
        metrics_updated = False
        if self.step_idx % self.metrics_frequency == 0:
            metrics, eval_gallery = self.get_ocr_metrics(sample)
            self.train_metrics |= metrics
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

    def get_ocr_metrics(self, sample):
        """
        In cruller_pretrain, this task returns some utils logs useful to monitor training.
        Typically, we want to return a few samples of images  and their generated OCR so
        that we can log them onto a tensorboard gallery in the log_step
        """
        metrics = {}
        eval_data = {}
        image_input, text_input, text_target = sample

        image_input = image_input.to(self.device_env.device, non_blocking=True)
        text_input = text_input[:, :-1].to(self.device_env.device, non_blocking=True)
        text_target = text_target[:, 1:].to(self.device_env.device, non_blocking=True)

        """
        metrics = {}
        image_input, text_input, text_target = sample
        text_input = [item[0] for item in text_input]
        text_input = torch.stack(text_input, dim=0).to(self.device_env.device, non_blocking=True)
        text_target = [item[0] for item in text_target]
        text_target = torch.stack(text_target, dim=0).to(self.device_env.device, non_blocking=True)
        image_input = image_input.to(self.device_env.device, non_blocking=True)

        # Add OCR-related metrics and generation

        ocr_metrics, _ = get_ocr_metrics(
            model=self.model,
            tokenizer=self.tokenizer,
            image_input=image_input,
            text_input=text_target,
            device_env=self.device_env,
            max_recursion_length=self.max_recursion_length,
        )"""

        # Add OCR-related metrics and generation

        ocr_metrics, ocr_reconstructed_sample = get_ocr_metrics(
            model=self.model,
            tokenizer=self.tokenizer,
            image_input=image_input,
            text_input=text_target,
            device_env=self.device_env,
            max_recursion_length=self.max_recursion_length,
            prompt_token=self.task_start_token,
        )
        if ocr_metrics and ocr_reconstructed_sample:
            metrics['ocr_reconstruction'] = ocr_metrics
            eval_data['ocr_reconstruction_data'] = ocr_reconstructed_sample
            print(ocr_metrics)
        else:
            _logger.info("Can't generate text from current batch. Skipping metrics...")
        
        # TODO Add other metrics relevant for eval step
        # 
        # metrics['metric_category'] = ... 
        return metrics, eval_data

    def __repr__(self):
        outputs = [
            f'model: {repr(self.model)}',
            f'opt: {repr(self.optimizer)}',
            f'sched: {repr(self.scheduler)}',
        ]
        return '\n'.join(outputs)

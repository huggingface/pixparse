import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import torchvision.transforms as transforms


from pixparse.framework import TaskEvalCfg, TaskEval, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerHF, TokenizerCfg
from pixparse.data import preprocess_text_anno
from pixparse.utils import get_ocr_metrics

from chug.common import LoaderBundle

_logger = logging.getLogger(__name__)

import time


@dataclass
class TaskCrullerEvalOCRCfg(TaskEvalCfg):
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


class TaskCrullerEvalOCR(TaskEval):
    """Cruller OCR evaluation Task
    * First evaluation task, pull out bits of code and refactor as needed over time
    * should be adaptable to several datasets as long as they are formatted correctly?
     Or do we impose datasets specific to this eval task? makes more sense maybe
    * Eval_step() is the singular method to accumulate metrics
    """

    def __init__(
        # Note, this initialization schema will be common for many tasks. how do I refactor it?
        self,
        cfg: TaskCrullerEvalOCRCfg,
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

        self.task_start_token = "<s_pretrain>"
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True
        self.tokenizer = TokenizerHF(cfg.tokenizer)
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )

        self.vocab_size = len(self.tokenizer.trunk)

        preproc_fn = preprocess_text_anno
        self.anno_preprocess_eval = partial(
            preproc_fn,
            tokenizer=self.tokenizer.trunk,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model)

        # We need to resize the token embeddings after the model has been initialized
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

        self.model.load_state_dict(cfg.model_state_dict)

        self.model.eval()

        self.has_no_sync = False
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_eval = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    cfg.model.image_encoder.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.CenterCrop(448),  # FIXME need better aspect preserving resize & pad
                transforms.Normalize(
                    # FIXME get mean / std from pretrained img model, fallback to 0.5 in random init
                    mean=(0.5,) * self.num_image_chs,
                    std=(0.5,) * self.num_image_chs,
                ),
            ]
        )

        # TODO These metrics have to be organized as dicts of dicts.
        # First level is the category, second level is the tag
        # We have to make this clear
        self.eval_metrics = {}
        self.max_recursion_length = 1000  # specific to Cruller for generation

    def time_and_log(func):
        """
        Method decorator to log execution time
        """
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            _logger.info(f"Executed method {func.__name__} in {execution_time:.2f} seconds")
            return result
        return wrapper
    
    def setup(self):
        device = self.device_env.device
        self.model.to(device)

    def prepare_for_evaluation(
        self, loaders: dict[str, LoaderBundle]
    ) -> dict[str, LoaderBundle]:
        loaders = {
            loader_key: loader
            for loader_key, loader in loaders.items()
            if loader_key in ["eval", "eval_FUNSD"]
        }

        return loaders
        # return loaders_and_tasks

    @time_and_log
    def step(self, sample):
        """
        Does one step of evaluation for OCR.
        """
        metrics = {}
        image_input, text_input, text_target = sample
        text_input = [item[0] for item in text_input]
        text_input = torch.stack(text_input, dim=0).to(
            self.device_env.device, non_blocking=True
        )
        text_target = [item[0] for item in text_target]
        text_target = torch.stack(text_target, dim=0).to(
            self.device_env.device, non_blocking=True
        )
        image_input = image_input.to(self.device_env.device, non_blocking=True)

        # Add OCR-related metrics and generation

        ocr_metrics, _ = get_ocr_metrics(
            model=self.model,
            tokenizer=self.tokenizer,
            image_input=image_input,
            text_input=text_target,
            device_env=self.device_env,
            max_recursion_length=self.max_recursion_length,
        )

        metrics["ocr_reconstruction"] = ocr_metrics

        # TODO Add other metrics relevant for eval step
        #
        # metrics['metric_category'] = ...
        return metrics

    def average_metrics(self, metrics: dict):
        wer_sum = 0
        cer_sum = 0
        for batch_metrics in metrics.values():
            wer_sum += batch_metrics["ocr_reconstruction"]["wer"]
            cer_sum += batch_metrics["ocr_reconstruction"]["cer"]

        num_batches = len(metrics)
        average_wer = wer_sum / num_batches
        average_cer = cer_sum / num_batches

        return {"ocr_reconstruction": {"wer": average_wer, "cer": average_cer}}

    def end(self):
        # process metrics, average them out? now done in self.average_metrics, called in evaluate, maybe end() should be called in evaluate 
        # TODO do that, call average_metrics in end
        pass

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        return state_dicts

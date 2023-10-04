import logging
from ast import literal_eval
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Lambda
from transformers import VisionEncoderDecoderModel

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from pixparse.data.loader import BaseCollate
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, TaskTrainCfg
from pixparse.models import ModelArgs, create_model
from pixparse.tokenizers import TokenizerCfg, create_tokenizer
from pixparse.utils.json_utils import json2token

_logger = logging.getLogger(__name__)


class CollateCORD(BaseCollate):
    r"""
    A collator for handling batches of PIL images and corresponding labels, as utilized with the CORDv2 dataset.
    Strings returned will be the tokenized json file matching the annotation.
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
    ):
        super().__init__(tokenizer, image_preprocess, start_token, max_length=max_length)

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        raw_texts = [literal_eval(item["ground_truth"])[
            "gt_parse"] for item in batch]
        labels_tokens = []
        for text in raw_texts:
            tokens_from_json, _ = json2token(
                text, self.tokenizer.all_special_tokens, sort_json_key=False
            )
            labels_tokens.append(
                self.tokenizer_fn(
                    self.start_token
                    # + self.tokenizer.bos_token
                    + tokens_from_json
                    + self.tokenizer.eos_token
                )
            )
        return self.pack_inputs(
            images,
            labels_tokens
        )


@dataclass
class TaskCrullerFinetuneCORDCfg(TaskTrainCfg):
    # override model default in spec per task
    model: ModelArgs = field(default_factory=lambda: ModelArgs(name="cruller_base",))

    def __post_init__(self):
        assert self.model.cfg is not None
        if self.tokenizer is None:
            # set tokenizer to text tower model name if not explicitly set
            self.tokenizer = TokenizerCfg(
                name=self.model.cfg.text_decoder.name)


def prepare_inputs_for_inference(
    tokenizer,
    input_ids: torch.Tensor,
    encoder_outputs: torch.Tensor,
    past_key_values=None,
    past=None,
    use_cache: bool = None,
    attention_mask: torch.Tensor = None,
):
    if past is not None:
        past_key_values = past
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "encoder_hidden_states": encoder_outputs,  # .last_hidden_state,
    }
    return output


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
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type, we may want to
        #  differentiate different precision modes such as amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (
                torch.bfloat16 if cfg.dtype in (
                    "bfloat16", "bf16") else torch.float16
            )
        model_cfg = self.cfg.model.cfg
        self.task_start_token = "<s_cord>"
        self.prompt_end_token = self.task_start_token
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
            "</s_service_price>",
            # This is only for CORD.
            # To reproduce it, check pixparse.utils.dataset_utils.get_additional_tokens_from_dataset
            "<s_subtotal_price>",
            "<s_discountprice>",
            "</s_sub>",
            "<s_sub>",
            "</s_total_etc>",
            "</s_discountprice>",
            "</s_vatyn>",
            "</s_subtotal_price>",
            "<s_changeprice>",
            "</s_total>",
            "</s_unitprice>",
            "<s_emoneyprice>",
            "</s_tax_price>",
            "</s_othersvc_price>",
            "</s_cnt>",
            "<s_vatyn>",
            "<s_unitprice>",
            "<s_total>",
            "<s_price>",
            "</s_price>",
            "<s_sub_total>",
            "</s_num>",
            "<s_total_etc>",
            "</s_creditcardprice>",
            "<s_tax_price>",
            "<s_menu>",
            "<s_nm>",
            "<s_menutype_cnt>",
            "</s_changeprice>",
            "<s_num>",
            "<s_itemsubtotal>",
            "</s_etc>",
            "<s_creditcardprice>",
            "</s_menuqty_cnt>",
            "</s_emoneyprice>",
            "<s_menuqty_cnt>",
            "<s_discount_price>",
            "</s_menu>",
            "</s_sub_total>",
            "<s_etc>",
            "</s_void_menu>",
            "<s_cashprice>",
            "</s_discount_price>",
            "</s_total_price>",
            "</s_nm>",
            "<s_service_price>",
            "<s_othersvc_price>",
            "</s_itemsubtotal>",
            "<s_void_menu>",
            "<s_total_price>",
            "</s_cashprice>",
            "</s_menutype_cnt>",
            "<s_cnt>",
        ]

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        """ Commenting out to test Donut weights
        """

        self.finetune_donut_weights = False
        _logger.info(
            f"Finetuning donut weights? {self.finetune_donut_weights}")

        if self.finetune_donut_weights:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "naver-clova-ix/donut-base")
        else:
            self.model = create_model(
                model_cfg,
                pretrained="",  # FIXME pass through tags or paths for full pretrained image-text tower
            )
            special_tokens_from_pretrain = [
                "<sep/>",  # JSON list separator
                "<s_pretrain>",  # task start (based on dataset/task)
            ]

            num_tokens_from_pretrain = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))})
            # need to resize embeddings from pretrained model in order to load it
            if num_tokens_from_pretrain > 0:
                self.model.text_decoder.trunk.resize_token_embeddings(
                    len(self.tokenizer)
                )

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        if self.finetune_donut_weights:
            self.num_image_chs = 3
        else:
            self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

        # TODO refactor, used in many tasks
        if self.finetune_donut_weights:
            img_mean = IMAGENET_DEFAULT_MEAN
            img_std = IMAGENET_DEFAULT_STD
        else:
            img_mean = self.model.image_encoder.trunk.pretrained_cfg["mean"]
            img_std = self.model.image_encoder.trunk.pretrained_cfg["std"]

        self.img_mean = (sum(img_mean) / len(img_mean) if cfg.model.image_encoder.image_fmt == "L" else img_mean)

        self.img_std = (sum(img_std) / len(img_std) if cfg.model.image_encoder.image_fmt == "L" else img_std)

        # preprocessors cross both the task/model & dataset domain, created within task here and passed to data loaders

        if self.finetune_donut_weights:
            image_size = (1280, 960)
            color_transform = Lambda(lambda x: x)
        else:
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
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense in setup here vs in __init__(). For
        # __init__ need the model structure to instantiate / setup tokenizer, other aspects. I don't think we need to
        # init weights / move to device until here if self.resume:
        if self.finetune_donut_weights:
            # We just add tokens, weights of donut are already initialized
            self.newly_added_num = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": sorted(set(self.special_tokens_finetune))})
            self.vocab_size = len(self.tokenizer)

            # We resize token embeddings after initializing
            if self.newly_added_num > 0:
                self.model.decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            _logger.info("Resuming from existing checkpoint.")
            self.state_dict = {k.replace("module.", ""): v for k, v in self.state_dict.items()}
            self.model.load_state_dict(self.state_dict)
            self.newly_added_num = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": sorted(
                    set(self.special_tokens_finetune))}
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
            # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend model size capacity beyond DDP
            # w/o overloading HF cluster NCCL throughput.
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device],
                static_graph=True,
            )
            self.has_no_sync = hasattr(self.model, "no_sync")

        self._setup_optimization(
            num_batches_per_interval=num_batches_per_interval,
        )

    def collate_fn(self, batch):
        return CollateCORD(
            self.tokenizer,
            self.image_preprocess_train,
            self.task_start_token,
        )

    def _forward(self, sample: Dict[str, Any]):
        image_input = sample["image"]
        label = sample["label"]
        text_target = sample["text_target"]
        with self.autocast():
            if self.finetune_donut_weights:
                output = self.model(pixel_values=image_input, decoder_input_ids=label, labels=text_target)
                logits = output["logits"]
            else:
                output = self.model(image_input, label)
                logits = output["logits"]

            loss = self.loss(
                logits.view(-1, self.vocab_size),
                text_target.view(-1),
            )

        if self.cfg.opt.grad_accum_steps > 1:
            loss /= self.cfg.opt.grad_accum_steps
        return loss

    def step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_input = sample["image"]
        label = sample["label"]
        text_target = sample["text_target"]
        result = {}
        image_input = image_input.to(self.device_env.device, non_blocking=True)
        label = label.to(self.device_env.device, non_blocking=True)
        text_target = text_target.to(self.device_env.device, non_blocking=True)

        need_update = (self.interval_batch_idx + 1) % self.cfg.opt.grad_accum_steps == 0

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
        state_dicts["tokenizer"] = (self.tokenizer.state_dict())
        # FIXME not needed anymore? we preprocess everything before
        return state_dicts

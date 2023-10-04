import logging

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from pixparse.framework import (DeviceEnv, Monitor, TaskEval, TaskEvalCfg)
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerCfg, TokenizerHF

_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerEvalRVLCDIPCfg(TaskEvalCfg):
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

class TaskCrullerEvalRVLCDIP(TaskEval):
    """Simple task to evaluate donut on FUNSD data and get metrics in similar format as Cruller."""

    def __init__(
        # Note, this initialization schema will be common for many tasks. how do I refactor it?
        self,
        cfg: TaskCrullerEvalRVLCDIPCfg,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (
                torch.bfloat16 if cfg.dtype in ("bfloat16", "bf16") else torch.float16
            )

        self.task_start_token = "<s_rvlcdip>"
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = TokenizerHF(cfg.tokenizer)

        self.state_dict = OrderedDict()
        self.resume = False

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

        self.int2str = {
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


        self.vocab_size = len(self.tokenizer.trunk)

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_eval = partial(
            preproc_fn,
            tokenizer=self.tokenizer.trunk,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        ) # UNUSED HERE

        self.model = Cruller(cfg.model)  # FIXME would be good to defer weight init here


        special_tokens_from_pretrain = [
                "<sep/>",  # JSON list separator
                "<s_pretrain>",  # task start (based on dataset/task)
            ]

        num_tokens_from_pretrain = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )
        # need to resize embeddings from pretrained model in order to load it
        if num_tokens_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

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
                    mean=self.img_mean,
                    std=self.img_std,
                ),
            ]
        )

    def setup(self):
        device = self.device_env.device

        self.newly_added_num = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(self.special_tokens_finetune))}
        )
        self.vocab_size = len(self.tokenizer.trunk)

        # We resize token embeddings after initializing
        if self.newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

        self.model.load_state_dict(self.resume_state_dict)
        self.model.eval()
        self.model.to(device)

    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past_key_values=None,
        past=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
    ):
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.trunk.pad_token_id).long()
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

    def prepare_for_evaluation(self, loaders):
        loaders = {
            loader_key: loader
            for loader_key, loader in loaders.items()
            if loader_key in ["eval", "eval_FUNSD"]
        }

        return loaders
        # return loaders_and_tasks


    def safe_image_transform(self, img):
        try:
            transformed_img = self.image_preprocess_eval(img)
        except PIL.UnidentifiedImageError as e:
            print(f'Encountered PIL issue {e}. Filtering...')
            transformed_img = None
        return transformed_img

    def collate_fn(self, batch):
        """
        basic collator for PIL images, as returned by rvlcdip dataloader (among others)
        """
        images = [item['image'] for item in batch if item is not None]
        labels = [item['label'] for item in batch if item is not None]

        if len(images) == 0:
            return None

        transformed_images = [self.safe_image_transform(img) for img in images]
        valid_indices = [i for i, img in enumerate(transformed_images) if img is not None]
        images = torch.stack([transformed_images[i] for i in valid_indices])
        labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.int64)

        return {'image': images, 'label': labels}

    def step(self, sample):
        metrics = {}
        metrics["classification"] = dict()
        with torch.inference_mode():
            image_outputs = self.model.image_encoder(sample["image"].to(self.device_env.device))
            correct_samples = 0
            for ground_truth, image_output in zip(sample["label"], image_outputs):
                current_string = "<s_rvlcdip>"
                input_ids = torch.tensor(self.tokenizer.trunk.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)
                max_steps = 5
                for _ in range(max_steps):
                    inputs = self.prepare_inputs_for_inference(input_ids=input_ids, encoder_outputs=image_output)
                    decoder_outputs = self.model.text_decoder(**inputs)

                    probabilities = F.softmax(decoder_outputs['logits'], dim=-1)
                    next_token_id = torch.argmax(probabilities[0, -1]).item()  # Just get the last token for the single sample

                    next_token = self.tokenizer.trunk.decode([next_token_id])
                    current_string += next_token

                    if next_token == "</s>":
                        generated_label = (current_string
                            .replace("<s_rvlcdip>", "")
                            .replace("</s>", "")
                            .replace("<s>", "")
                            .strip()
                        )
                        ground_truth_label = "<" + self.int2str[int(ground_truth)] + "/>"
                        if generated_label == ground_truth_label:
                            correct_samples += 1
                        break

                    input_ids = torch.tensor(self.tokenizer.trunk.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)
        metrics["classification"]["correct_samples"] = correct_samples
        metrics["classification"]["n_valid_samples"] = len(sample['label'])
        return metrics

    def average_metrics(self, metrics: dict):
        correct_samples = 0
        total_samples = 0 # Because of invalid samples, metrics might be slightly different so we keep track of each # of valid samples / batch
        for batch_metrics in metrics.values():
            correct_samples += batch_metrics["classification"]["correct_samples"]
            total_samples += batch_metrics["classification"]["n_valid_samples"]

        average_acc = correct_samples / total_samples

        return {"classification": {"accuracy": average_acc}}

    def end(self):
        # process metrics, average them out? now done in self.average_metrics, called in evaluate, maybe end() should be called in evaluate
        # TODO do that, call average_metrics in end
        pass

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        return state_dicts

import logging

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import PIL
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from pixparse.framework import (DeviceEnv, Monitor, TaskEval, TaskEvalCfg)
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerCfg, TokenizerHF
from pixparse.utils.json_utils import json2token, token2json
from pixparse.utils.json_utils import JSONParseEvaluator

import numpy as np

from ast import literal_eval

_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerEvalDOCVQACfg(TaskEvalCfg):
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

class TaskCrullerEvalDOCVQA(TaskEval):
    """Simple task to evaluate donut on FUNSD data and get metrics in similar format as Cruller."""

    def __init__(
        # Note, this initialization schema will be common for many tasks. how do I refactor it?
        self,
        cfg: TaskCrullerEvalDOCVQACfg,
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

        self.task_start_token = "<s_docvqa>"
        self.prompt_end_token = "<answer/>"
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = TokenizerHF(cfg.tokenizer)

        self.state_dict = OrderedDict()
        self.resume = False
        docvqa_finetune_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
            # ... ADD DOCVQA TOKENS
        ]

        # ---- add pretraining tokens 
        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_eval = partial(
            preproc_fn,
            tokenizer=self.tokenizer.trunk,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model)  # FIXME would be good to defer weight init here

        # ---- Add pretraining tokens

        newly_added_num_from_pretrain = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )


        # need to resize embeddings from pretrained model in order to load it
        if newly_added_num_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

        # ---- Add finetuning tokens

        newly_added_num = self.tokenizer.trunk.add_special_tokens(
            {"additional_special_tokens": sorted(set(docvqa_finetune_tokens))}
        )
        self.vocab_size = len(self.tokenizer.trunk)

        # We resize token embeddings after initializing
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer.trunk)
            )

# ------ 
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

        # TODO refactor, used in many tasks

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
                transforms.Grayscale(),
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
        self.model.load_state_dict(self.resume_state_dict)
        self.model.eval()
        self.model.to(device)
        self.all_ground_truths = []
        self.all_predictions = []
        self.acc_list = []

        self.evaluator = JSONParseEvaluator()


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

    def text_input_to_target(self, text_input, ignore_id=-100):
        target = text_input.clone()
        # model doesn't need to predict pad token
        target[target == self.tokenizer.trunk.pad_token_id] = ignore_id
        # model doesn't need to predict prompt (for VQA)
        prompt_end_token_id = self.tokenizer.trunk.convert_tokens_to_ids(
            self.prompt_end_token
        )
        slice_id = torch.nonzero(target == prompt_end_token_id).sum() + 1
        target[:slice_id] = ignore_id
        return target

    def collate_fn(self, batch):
        images = [item['image'] for item in batch]
        q_and_as = [np.random.choice(item['labels']) for item in batch]
    
        transform = self.image_preprocess_train
        images = torch.stack([transform(img) for img in images])
    
        return {
            "image": images,
            "prompt": q_and_as,
        }


    def step(self, batch):
        """
        Does one step of evaluation for DOCVQA. 
        Current limitation: sample-by-sample decoding.
        """
        metrics = {}
        for image, prompt in zip(batch["image"], batch['prompt']):
            decoded_gt = self.tokenizer.trunk.decode(prompt)
            ground_truth = token2json(decoded_gt)
            with torch.inference_mode():
                tensor_image = image.unsqueeze(0).to(self.device_env.device)  # Adding an extra dimension for batch
                output = self.model.image_encoder(tensor_image)
                
                # split out answer from prompt

                current_string = self.task_start_token + prompt.split("<s_answer>")[0] + "<s_answer>" 
                input_ids = torch.tensor(self.tokenizer.trunk.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)  # Adding extra dimension for batch
                max_steps = 512  # maximum number of steps

                for step in range(max_steps):
                    inputs = self.prepare_inputs_for_inference(input_ids=input_ids, encoder_outputs=output)
                    
                    decoder_outputs = self.model.text_decoder(**inputs)
                    
                    probabilities = F.softmax(decoder_outputs['logits'], dim=-1)
                    next_token_id = torch.argmax(probabilities[0, -1]).item()  # Just get the last token for the single sample
                    
                    next_token = self.tokenizer.trunk.decode([next_token_id])
                    current_string += next_token

                    if next_token == "</s>":
                        break

                    input_ids = torch.tensor(self.tokenizer.trunk.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)

                predicted_json = token2json(current_string)

            # FIXME we should only compare generated answers vs ground truth answers
            self.all_predictions.append(predicted_json)
            self.all_ground_truths.append(ground_truth)
            acc = self.evaluator.cal_acc(predicted_json, ground_truth)
            self.acc_list.append(acc)

        metrics["batch_accuracy"] = acc 
        return metrics

    def average_metrics(self, metrics: dict):
        avg_accuracy = np.mean(self.acc_list)      
        f1 = self.evaluator.cal_f1(self.all_predictions, self.all_ground_truths)

        self.all_ground_truths = []
        self.all_predictions = []
        self.acc_list = []

        return {"average_accuracy": avg_accuracy, "f1_score": f1}

    def end(self):
        # process metrics, average them out? now done in self.average_metrics, called in evaluate, maybe end() should be called in evaluate
        # TODO do that, call average_metrics in end
        pass

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        return state_dicts

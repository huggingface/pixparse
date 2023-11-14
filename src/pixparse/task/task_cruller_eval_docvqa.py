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
import numpy as np

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno, create_transforms
from pixparse.data.loader import BaseCollate
from pixparse.framework import (DeviceEnv, Monitor, TaskEval, TaskEvalCfg)
from pixparse.models import Cruller, ModelCfg, get_model_config, ModelArgs, create_model, resize_model_embeddings
from pixparse.tokenizers import TokenizerCfg, create_tokenizer
from pixparse.utils.json_utils import json2token, token2json
from pixparse.utils.json_utils import JSONParseEvaluator
from pixparse.utils.metrics import average_normalized_levenshtein_similarity


_logger = logging.getLogger(__name__)



@dataclass
class TaskCrullerEvalDOCVQACfg(TaskEvalCfg):
    model: ModelArgs = field(default_factory=lambda: ModelArgs(
        name='cruller_base',  # override model default in spec per task
    ))

    def __post_init__(self):
        assert self.model.cfg is not None
        if self.tokenizer is None:
            # set tokenizer to text tower model name if not explicitly set
            self.tokenizer = TokenizerCfg(name=self.model.cfg.text_decoder.name)


class TaskCrullerEvalDOCVQA(TaskEval):
    """Simple task to evaluate donut on FUNSD data and get metrics in similar format as Cruller."""

    def __init__(
        self,
        cfg: TaskCrullerEvalDOCVQACfg,
        checkpoint_path: str,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        model_cfg = self.cfg.model.cfg
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = (
                torch.bfloat16 if cfg.dtype in ("bfloat16", "bf16") else torch.float16
            )

        self.task_start_token = "<s_docvqa>"
        self.prompt_end_token = "<s_answer>"
        self.max_position_embeddings = model_cfg.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments
        self.tokenizer = create_tokenizer(cfg.tokenizer)

        special_tokens_from_pretrain = [
            "<sep/>",  # JSON list separator
            "<s_pretrain>",  # task start (based on dataset/task)
        ]
        
        num_pretrain_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(additional_special_tokens))}
        )

        finetuning_special_tokens = [
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
            "<s_question>",
            "</s_question>",
            "</s_answer>",
        ]
        additional_special_tokens = special_tokens_from_pretrain + finetuning_special_tokens 

        num_finetuning_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(additional_special_tokens))}, replace_additional_special_tokens=False
        )
        
                
        self.model = create_model(
            model_cfg,
            pretrained=checkpoint_path, 
            num_new_tokens=num_pretrain_tokens + num_finetuning_tokens
        )


        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_eval = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )
        self.model = create_model(
            model_cfg,
            pretrained='',
        )
        # ---- Add pretraining tokens

        newly_added_num_from_pretrain = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens_from_pretrain))}
        )
        # need to resize embeddings from pretrained model in order to load it
        if newly_added_num_from_pretrain > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer)
            )

        # ---- Add finetuning tokens

        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(docvqa_finetune_tokens))}
        )
        self.vocab_size = len(self.tokenizer)
        # We resize token embeddings after initializing
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(
                len(self.tokenizer)
            )
        # ------ Load checkpoint (debug)
        state_dict = torch.load(checkpoint_path)['model']
        eval_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(eval_state_dict)

        self.has_no_sync = False

        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.image_preprocess_eval = create_transforms(
            self.cfg.image_transforms,
            input_cfg=self.image_input_cfg,
            training=False,
            interpolation='bicubic',
            crop_margin=False,  # True?
            align_long_axis=False,
        )
        self.raw_predictions_test = dict()

    def setup(self):
        device = self.device_env.device
        self.model.to(device)
        self.model.eval()
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
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
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

    def collate_fn(self, batch):
        question_ids = []
        image_ids = []
        images = []
        questions = []
        answers = []
        for item in batch:
            question_ids.append(item['question_id'])
            image_ids.append(item['image_id'])
            images.append(item['image'])
            questions.append(item['labels']["question"])
            answers.append(item['labels']["answers"])
        transform = self.image_preprocess_eval
        images = torch.stack([transform(img) for img in images])
        return {
            "images": images,
            "questions": questions,
            "ground_truth_answers": answers,
            "image_ids": image_ids,
            "question_ids": question_ids,
        }

    def step(self, batch):
        """
        Does one step of evaluation for DOCVQA. 
        Current limitation: sample-by-sample decoding.
        """
        metrics = {}
        image_outputs = self.model.image_encoder(batch['images'].to(self.device_env.device))
        for output, question, answers, question_id in zip(image_outputs, batch['questions'], batch['ground_truth_answers'], batch['question_ids']):
            self.all_ground_truths.append(answers)
            with torch.inference_mode():
                # split out answer from prompt
                current_string = self.task_start_token + "<s_question>" + question + "</s_question>" + "<s_answer>" 
                input_ids = torch.tensor(self.tokenizer.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)  # Adding extra dimension for batch
                max_steps = 512  # maximum number of steps

                for generation_step in range(max_steps):
                    inputs = self.prepare_inputs_for_inference(input_ids=input_ids, encoder_outputs=output)
                    decoder_outputs = self.model.text_decoder(**inputs)
                    probabilities = F.softmax(decoder_outputs['logits'], dim=-1)
                    next_token_id = torch.argmax(probabilities[0, -1]).item()  # Just get the last token for the single sample
                    next_token = self.tokenizer.decode([next_token_id])
                    current_string += next_token
                    if next_token == "</s>":
                        break
                    input_ids = torch.tensor(self.tokenizer.encode(current_string, add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)
                predicted_json = token2json(current_string)
            if 'answer' in predicted_json:
                self.all_predictions.append(predicted_json['answer'])
            else:
                self.all_predictions.append("")
        return metrics

    def average_metrics(self, metrics: dict):
        anls = average_normalized_levenshtein_similarity(ground_truth=self.all_ground_truths, predicted_answers=self.all_predictions)
        return {"ANLS": anls}

    def prepare_for_evaluation(
        self, loaders
    ):
        loaders = {
            loader_key: loader
            for loader_key, loader in loaders.items()
            if loader_key in ["eval", "eval_FUNSD"]
        }

        return loaders

    def end(self):
        # process metrics, average them out? now done in self.average_metrics, called in evaluate, maybe end() should be called in evaluate
        # TODO do that, call average_metrics in end
        pass

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        return state_dicts

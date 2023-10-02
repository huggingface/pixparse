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
from torchvision.transforms import Lambda

from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from pixparse.framework import DeviceEnv, Monitor, TaskEval, TaskEvalCfg
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerCfg, TokenizerHF
from pixparse.utils.json_utils import json2token, token2json
from pixparse.utils.json_utils import JSONParseEvaluator
from pixparse.utils.metrics import average_normalized_levenshtein_similarity

from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import (
    AutoTokenizer,
    MBartConfig,
    MBartForCausalLM,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    MBartTokenizer,
    BartTokenizer,
    BartTokenizerFast
)


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import time
import os
import json

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


class DonutTokenizer(nn.Module):
    def __init__(self):
        super().__init__()

        """self.trunk = XLMRobertaTokenizer.from_pretrained(
                "hyunwoongko/asian-bart-ecjk"
            )
        """


        """
        self.trunk = XLMRobertaTokenizerFast.from_pretrained(
                "hyunwoongko/asian-bart-ecjk"
            ) # handles spaces sort of, and adds weird tokens ion>What is the date when the approval form was filled ?</s_question><s_answer>▁june▁7,▁1988ko_KR</s>
            # off by one error?

        """



        self.trunk = XLMRobertaTokenizerFast.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-docvqa"
            ) # Good answers but misses whitespaces, probably same as AutoTokenizer


        #self.trunk = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa") # Good answers but misses whitespaces
        #self.trunk = MBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk") # adds a bunch of de_DE ko_KR, etc
        #self.trunk = BartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk") # NOT WORKING

def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Resize position embeddings
    Truncate if sequence length of Bart backbone is greater than given max_length,
    else interpolate to max_length
    """
    breakpoint()
    if weight.shape[0] > max_length:
        weight = weight[:max_length, ...]
    else:
        weight = (
            F.interpolate(
                weight.permute(1, 0).unsqueeze(0),
                size=max_length,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 0)
        )
    breakpoint()
    return weight

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
        self.prompt_end_token = "<s_answer>"
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = True  # set for image-text dataset experiments

        self.resize_embeddings = False
        self.max_length = 128
        self.eval_donut = False

        if self.eval_donut:
            self.tokenizer = DonutTokenizer()
        else:
            self.tokenizer = TokenizerHF(cfg.tokenizer)

        self.state_dict = OrderedDict()
        self.resume = False



        if self.eval_donut:
            docvqa_finetune_tokens = [
                "</s_answer>",
                "</s_question>",
                "<no/>",
                "<s_answer>",
                "<s_docvqa>",
#                "<s_iitcdip>",
                "<s_question>",
#                "<s_synthdog>",
                "<yes/>",
            ]
        else:
            docvqa_finetune_tokens = [
                "<sep/>",  # JSON list separator
                self.task_start_token,  # task start (based on dataset/task)
                self.prompt_end_token,  # prompt end (or task_start for pretrain)
                "<s_question>",
                "</s_question>",
                "</s_answer>",
#                "<yes/>",
#                "<no/>",
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

        if self.eval_donut:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-docvqa"
            )

            newly_added_num = self.tokenizer.trunk.add_special_tokens(
                {"additional_special_tokens": sorted(set(docvqa_finetune_tokens))}
            )
            self.vocab_size = len(self.tokenizer.trunk)

            # We resize token embeddings after initializing
            if newly_added_num > 0:
                self.model.decoder.resize_token_embeddings(len(self.tokenizer.trunk))


            # move around positional embeddings depending on max_length
            if self.resize_embeddings:  # if max_length of trained model differs max_length you want to train
                self.model.decoder.model.decoder.embed_positions.weight = torch.nn.Parameter(
                    resize_bart_abs_pos_emb(
                        self.model.decoder.model.decoder.embed_positions.weight,
                        self.max_length
                        + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                    )
                )
                self.max_position_embeddings = self.max_length
        else:
            self.model = Cruller(
                cfg.model
            )  # FIXME would be good to defer weight init here
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


            if self.resize_embeddings:  # if max_length of trained model differs max_length you want to train
                self.model.text_decoder.trunk.model.decoder.embed_positions.weight = torch.nn.Parameter(
                    resize_bart_abs_pos_emb(
                        self.model.text_decoder.trunk.model.decoder.embed_positions.weight,
                        self.max_length
                        + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                    )
                )
                self.max_position_embeddings = self.max_length

        # ------
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        if self.eval_donut:
            self.num_image_chs = 3
        else:
            self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == "L" else 3

        # TODO refactor, used in many tasks
        if self.eval_donut:
            img_mean = IMAGENET_DEFAULT_MEAN
            img_std = IMAGENET_DEFAULT_STD
        else:
            img_mean = self.model.image_encoder.trunk.pretrained_cfg["mean"]
            img_std = self.model.image_encoder.trunk.pretrained_cfg["std"]

        self.img_mean = (
            sum(img_mean) / len(img_mean)
            if cfg.model.image_encoder.image_fmt == "L" and not self.eval_donut
            else img_mean
        )
        self.img_std = (
            sum(img_std) / len(img_std)
            if cfg.model.image_encoder.image_fmt == "L" and not self.eval_donut
            else img_std
        )

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        if self.eval_donut:
            image_size = (2560, 1920)
            color_transform = Lambda(lambda x: x)
        else:
            image_size = cfg.model.image_encoder.image_size
            color_transform = transforms.Grayscale()
        print(self.img_mean, self.img_std)
        self.image_preprocess_eval = transforms.Compose(
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

        self.raw_predictions_test = dict()

    def setup(self):
        device = self.device_env.device
        if self.eval_donut:
            pass  # We directly load the model we want to evaluate
        else:
            if not self.resize_embeddings:
                # load the state dict as is
                self.model.load_state_dict(self.resume_state_dict)
            else:
                breakpoint()
                initial_state_dict = self.model.state_dict()
                new_state_dict = self.resume_state_dict
                for x in new_state_dict:
                    if x.endswith("embed_positions.weight") and self.max_position_embeddings != 1024:
                        breakpoint()
                        new_state_dict[x] = torch.nn.Parameter(
                            resize_bart_abs_pos_emb(
                                initial_state_dict[x],
                                self.max_position_embeddings
                                + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                            )
                        )
                    elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                        new_state_dict[x] = initial_state_dict[x][: len(self.tokenizer.trunk), :]
                    else:
                        new_state_dict[x] = initial_state_dict[x]
                breakpoint()
                self.model.load_state_dict(new_state_dict)




        self.model.eval()
        self.model.to(device)
        self.all_ground_truths = []
        self.all_predictions = []
        self.test_set_answers = []
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
            print(f"Encountered PIL issue {e}. Filtering...")
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

    def time_and_log(func):
        """
        Method decorator to log execution time
        """

        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            _logger.info(
                f"Executed method {func.__name__} in {execution_time:.2f} seconds"
            )
            return result

        return wrapper

    def collate_fn(self, batch):
        question_ids = []
        image_ids = []
        images = []
        questions = []
        answers = []
        for item in batch:
            question_ids.append(item["question_id"])
            image_ids.append(item["image_id"])
            images.append(item["image"])
            questions.append(item["labels"]["question"])
            answers.append(item["labels"]["answers"])

        transform = self.image_preprocess_eval
        images = torch.stack([transform(img) for img in images])

        return {
            "images": images,
            "questions": questions,
            "ground_truth_answers": answers,
            "image_ids": image_ids,
            "question_ids": question_ids,
        }

    @time_and_log
    def step(self, batch):
        """
        Does one step of evaluation for DOCVQA.
        Current limitation: sample-by-sample decoding.
        """
        metrics = {}
        if self.eval_donut:
            image_input = batch["images"]
            image_outputs = self.model.encoder(image_input.to(self.device_env.device))
            #rgb_batch_tensor = image_input.expand(-1, 3, -1, -1)
            #image_outputs = self.model.encoder(
            #    rgb_batch_tensor.to(self.device_env.device)
            #)
            image_outputs = image_outputs.last_hidden_state
        else:
            image_outputs = self.model.image_encoder(
                batch["images"].to(self.device_env.device)
            )
        for output, question, answers, question_id in zip(
            image_outputs,
            batch["questions"],
            batch["ground_truth_answers"],
            batch["question_ids"],
        ):
            self.all_ground_truths.append(answers)
            with torch.inference_mode():
                # split out answer from prompt
                current_string = (
                    self.task_start_token
                    + "<s_question>"
                    + question
                    + "</s_question>"
                    + "<s_answer>"
                )
                input_ids = self.tokenizer.trunk.encode(current_string, add_special_tokens=False, return_tensors="pt").to(self.device_env.device)
                # Adding extra dimension for batch
                # generate output
                max_steps = 128  # maximum number of steps


                for step in range(max_steps):
                    inputs = self.prepare_inputs_for_inference(
                        input_ids=input_ids, encoder_outputs=output
                    )
                    if self.eval_donut:
                        decoder_outputs = self.model.decoder(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            encoder_hidden_states=inputs["encoder_hidden_states"],
                            past_key_values=None,
                            use_cache=None,
                            return_dict=True,
                        )
                    else:
                        decoder_outputs = self.model.text_decoder(**inputs)

                    probabilities = F.softmax(decoder_outputs["logits"], dim=-1)
                    next_token_id = torch.argmax(
                        probabilities[0, -1]
                    ).item()  # Just get the last token for the single sample

                    next_token = self.tokenizer.trunk.decode([next_token_id])
                    current_string += next_token
                    print(current_string)

                    if next_token == "</s>":
                        break

                    input_ids = self.tokenizer.trunk.encode(current_string, add_special_tokens=False, return_tensors="pt").to(self.device_env.device)

                predicted_json = token2json(current_string)
                if "answer" in predicted_json:
                    self.all_predictions.append(predicted_json["answer"])
                    self.test_set_answers.append(
                        {"questionId": question_id, "answer": predicted_json["answer"]}
                    )
                else:
                    self.all_predictions.append("")
                    self.test_set_answers.append(
                        {"questionId": question_id, "answer": ""}
                    )
        return metrics

    def average_metrics(self, metrics: dict):
        anls = average_normalized_levenshtein_similarity(
            ground_truth=self.all_ground_truths, predicted_answers=self.all_predictions
        )
        # FIXME should only save for test set
        # FIXME pass along path from config and experiment state
        with open(
            "/fsx/pablo/metrics_docvqa/vqa_swin_384_to_1920_testset_train+val_trained_samecount.json", "w"
        ) as f:
            json.dump(self.test_set_answers, f)
        return {"ANLS": anls}

    def end(self):
        # process metrics, average them out? now done in self.average_metrics, called in evaluate, maybe end() should be called in evaluate
        # TODO do that, call average_metrics in end
        pass

    def state_dict(self):
        state_dicts = {}
        state_dicts["model"] = self.model.state_dict()
        return state_dicts

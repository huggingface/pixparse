import matplotlib.pyplot as plt
import os
import torch
import pixparse
import chug
import webdataset as wds
import re
import json
import pandas as pd
import editdistance
from pixparse.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno
from pixparse.app.train import TaskCrullerPretrainCfg
from PIL import Image
from transformers import AutoTokenizer
import torch.nn.functional as F
## OCR pretraining metrics


def calculate_cer(reference, hypothesis):
    """
    Calculation of CER with Levenshtein distance.

    :param reference: reference text string
    :param hypothesis: recognized text string
    :return: CER
    """
    ref = reference.replace(' ', '')  # remove space
    hyp = hypothesis.replace(' ', '')  # remove space

    cer = editdistance.eval(ref, hyp) / len(ref)
    return cer

def calculate_wer(reference, hypothesis):
    """
    Calculation of WER with Levenshtein distance.

    :param reference: reference text string
    :param hypothesis: recognized text string
    :return: WER
    """
    ref_words = reference.split(' ')
    hyp_words = hypothesis.split(' ')

    wer = editdistance.eval(ref_words, hyp_words) / len(ref_words)
    return wer



cfg = TaskCrullerPretrainCfg(model_name="cruller_base")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_decoder.name)
special_tokens = [
            "<sep/>",  # JSON list separator
            '<s_pretrain>',  # task start (based on dataset/task)
            '<s_pretrain>',  # prompt end (or task_start for pretrain)
        ]
newly_added_num = tokenizer.add_special_tokens(
    {"additional_special_tokens": sorted(set(special_tokens))}
)


with torch.inference_mode():
    image_encoding = model.image_encoder(tensor_img)

def prepare_inputs_for_inference(
        model,
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
            "encoder_hidden_states": encoder_outputs, #.last_hidden_state,
        }
        return output

def get_next_token(next_token_logits, use_sample=True, temperature=5):
    if use_sample:
        relevant_logits = next_token_logits / temperature
        probs = F.softmax(relevant_logits, dim=-1)

        next_token_id = torch.multinomial(
            probs, num_samples=1
        ).reshape(-1).unsqueeze(-1)
    else:
        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
    return next_token_id, probs



image_path = "./FUNSD_dataset/dataset/training_data/images/0000971160.png"
num_image_chs = 1
image = Image.open(image_path).convert('L')
# Preprocessing like Cruller pretraining
image_preprocess_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                cfg.model.image_encoder.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True),
            #transforms.CenterCrop(448),  # FIXME need better aspect preserving resize & pad
            transforms.Normalize(
                # FIXME get mean / std from pretrained img model, fallback to 0.5 in random init
                mean=(0.5,) * num_image_chs,
                std=(0.5,) * num_image_chs,
            )
        ])
tensor_img = torch.unsqueeze(image_preprocess_train(image), 0)
encoder_outputs = image_encoding

input_ids = torch.tensor([tokens_prompt['input_ids'][:-2] + [tokenizer('<s_pretrain>')['input_ids'][1]]])
recursion_length = 0
prob_arr = []
with torch.inference_mode():
    while True:
        inputs = prepare_inputs_for_inference(model.text_decoder, tokenizer, input_ids=input_ids, encoder_outputs=encoder_outputs)
        outputs = model.text_decoder.forward(**inputs)

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id, probs = get_next_token(next_token_logits)
        prob_arr.append(probs.numpy())

        if next_token_id.item() == tokenizer.eos_token_id or recursion_length > 5:
            break
        recursion_length += 1
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
generated_text = tokenizer.decode(input_ids[0])


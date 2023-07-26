from pixparse.utils.ocr_utils import get_cer_wer_metrics 
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from torch import nn

from jiwer import cer, wer
import jiwer.transforms as tr

import re
from typing import List, Optional
import csv

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

        ocr_metrics, _ = get_donut_ocr_metrics(
            model=self.model,
            tokenizer=self.tokenizer,
            image_input=image_input,
            text_input=text_target,
            device_env=self.device_env,
            max_recursion_length=self.max_recursion_length,
        )


"""


def get_donut_ocr_metrics(
        donut_model: VisionEncoderDecoderModel,
        donut_processor: DonutProcessor,
        image_input,
        text_input,
        device_env,
        max_recursion_length
) -> dict:
    cer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.Strip(),
            tr.ReduceToListOfListOfChars(),
        ]
    )

    wer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
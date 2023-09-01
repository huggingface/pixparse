from PIL import Image
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from dataclasses import dataclass
from functools import partial


from pixparse.framework import TaskEvalCfg, TaskEval, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.data import preprocess_text_anno
from pixparse.utils import get_ocr_metrics
from pixparse.utils.ocr_utils import get_cer_wer_metrics

import jiwer.transforms as tr

import torch
import torchvision.transforms as transforms

import numpy as np

@dataclass
class TaskDonutEvalOCRCfg(TaskEvalCfg):
    def __post_init__(self):
        pass

class TaskDonutEvalOCR(TaskEval):
    """Simple task to evaluate donut on FUNSD data and get metrics in similar format as Cruller.
    """

    def __init__(
        # Note, this initialization schema will be common for many tasks. how do I refactor it?
        self,
        cfg: TaskDonutEvalOCRCfg,
        device_env: DeviceEnv,
        monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.task_prompt = "<s_cord-v2>"
        self.decoder_input_ids = self.processor.tokenizer(self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        self.vocab_size = len(self.processor.tokenizer)

        preproc_fn = preprocess_text_anno
        self.max_position_embeddings = 768
        self.anno_preprocess_eval = partial(
            preproc_fn,
            tokenizer=self.processor.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token="",
            prompt_end_token=self.task_prompt,
        )

        self.model.eval()

        self.has_no_sync = False
        self.num_image_chs = 3 # 3

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_eval = lambda x: x 
        self.cer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.Strip(),
            tr.ReduceToListOfListOfChars(),
        ]
    )

        self.wer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
        self.eval_metrics = {}
        self.max_recursion_length = 1000  # specific to Cruller for generation

    
    def setup(self):
        device = self.device_env.device
        self.model.to(device)

    def prepare_for_evaluation(
        self, loaders
    ):
        loaders = {
            loader_key: loader
            for loader_key, loader in loaders.items()
            if loader_key in ["eval", "eval_FUNSD"]
        }

        return loaders
        # return loaders_and_tasks

    def clean_text(self, text: str) -> str:
        sequence = text.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        cleaned_text = re.sub('<.*?>', '', sequence)
        return cleaned_text
    
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

        # Compute OCR metrics for Donut

        decoder_input_ids = self.processor.tokenizer(self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = self.processor([im.convert('RGB') for im in image_input], return_tensors="pt").pixel_values

        with torch.inference_mode():
            outputs = [self.model.generate(
                pixel_value.unsqueeze(0).to(self.device_env.device),
                decoder_input_ids=decoder_input_ids.to(self.device_env.device),
                max_length=self.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            ) for pixel_value in pixel_values]
        generated_text = [self.clean_text(self.processor.decode(greedy_outputs.sequences[0])) for greedy_outputs in outputs]
        text_input[
            text_input == -100
        ] = (
            self.processor.tokenizer.pad_token_id
        ) 
        raw_decoded_texts = self.processor.tokenizer.batch_decode(text_input)
        decoded_texts = [self.clean_text(t) for t in raw_decoded_texts]
        # FIXME sometimes we are decoding no text at all after cleaning
        filtered = [
            (ref, pred)
            for ref, pred in zip(decoded_texts, generated_text)
            if ref and pred
        ]

        if not filtered:
            return None, None

        decoded_texts, ocr_predictions = zip(*filtered)

        decoded_texts = list(decoded_texts)
        ocr_predictions = list(ocr_predictions)
        
        ocr_predictions = [
            text[0 : len(reftext)]
            for text, reftext in zip(ocr_predictions, decoded_texts)
        ]

        metrics["ocr_reconstruction"] = get_cer_wer_metrics(
            self.cer_transforms,
            self.wer_transforms,
            dict(),
            ocr_predictions,
            decoded_texts,
        )
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

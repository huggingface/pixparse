from PIL import Image
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from dataclasses import dataclass

from pixparse.framework import TaskEvalCfg, TaskEval, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.tokenizers import TokenizerHF, TokenizerCfg
from pixparse.data import preprocess_text_anno
from pixparse.utils import get_ocr_metrics

@dataclass
class TaskDonutEvalOCRCfg(TaskEvalCfg):
    def __post_init__(self):
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.task_prompt = "<s_cord-v2>"
        self.decoder_input_ids = self.processor.tokenizer(self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids


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


        # Compute OCR metrics for Donut


        decoder_input_ids = self.processor.tokenizer([self.task_prompt] * self.cfg.batch_size, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = self.processor(image_input, return_tensors="pt").pixel_values
        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = self.processor.batch_decode(outputs.sequences)[0] #FIXME only first sentence taken
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        generated_text = re.sub(r"<.*?>", "", sequence, count=1).strip() # FIXME not passed along

        decoded_texts = self.tokenizer.trunk.batch_decode(text_input)
        ocr_predictions = [
            re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in ocr_predictions
        ]
        decoded_texts = [
            re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in decoded_texts
        ]

        # FIXME sometimes we are decoding no text at all after cleaning
        filtered = [
            (ref, pred)
            for ref, pred in zip(decoded_texts, ocr_predictions)
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

        ocr_pretraining_metrics = get_cer_wer_metrics(
            cer_transforms,
            wer_transforms,
            ocr_pretraining_metrics,
            ocr_predictions,
            decoded_texts,
        )
        reconstructed_sample = {
            "image": image_input[0],
            "original_text": decoded_texts[0],
            "reconstructed_text": ocr_predictions[0],
        }
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

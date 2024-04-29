from pixparse.models import ModelArgs, create_model
from pixparse.models.config import ModelCfg
from pixparse.utils import get_ocr_metrics
from pixparse.tokenizers import create_tokenizer, TokenizerCfg
from PIL import Image
from pixparse.framework import DeviceEnv
import torch
from torchvision import transforms
from typing import  Union
from jiwer import cer, wer
import jiwer.transforms as tr
import re
class SmallEval():
    def __init__(
        self,
        cfg:  Union[str, ModelCfg] ,
        checkpoint_path: str ,
        device_env: DeviceEnv
        ):

        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
       #model_cfg = get_model_config(self.cfg)
        self.cfg = ModelCfg.load(cfg)
        cfg_tokenizer = TokenizerCfg(name=self.cfg.text_decoder.name)
        self.tokenizer = create_tokenizer(cfg_tokenizer)
        #create_tokenizer(cfg.tokenizer)
        self.task_start_token = "<s_pretrain>"
        self.prompt_end_token = self.task_start_token
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )
        self.vocab_size = len(self.tokenizer)
        self.model = create_model(self.cfg, new_vocab_size=self.vocab_size, pretrained=self.checkpoint_path)
        self.image_input_cfg = self.model.image_encoder.traits.get('input')
        self.max_position_embeddings = self.cfg.text_decoder.max_length
        self.text_anno_fn = True
        
        self.device_env = device_env

    def setup(self):
        """
        Weight initialization is deferred here. The state_dict to be loaded has to be created before, within the task.
        """
       # device = self.device_env.device
        self.model.eval()
        self.model.to(self.device_env.device)

    def prepare_image_output(self, img_path: str):
        #use self.image_preprocess_eval to prepare image
        img_mean = self.model.image_encoder.trunk.pretrained_cfg["mean"]
        img_std = self.model.image_encoder.trunk.pretrained_cfg["std"]

        img_mean = (
        sum(img_mean) / len(img_mean)
    #if taskcfg.model.image_encoder.image_fmt == "L"
    #else img_mean 
        )
        img_std = (
            sum(img_std) / len(img_std)
            #if taskcfg.model.image_encoder.image_fmt == "L"
            #else img_std
        )
        image = Image.open(img_path).convert("L")
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (1280, 960),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.Normalize(
            mean=img_mean,
            std=img_std,
        ),])
        image = transform(image).to(self.device_env.device)
        image = image.unsqueeze(0)
        output = self.model.image_encoder(image) 
    # current_string = (task_prompt + "<s_question>" + user_input + "</s_question><s_answer>")
        return output
    
    def get_preds(self, image_output):
        input_ids = (
            torch.tensor(
                self.tokenizer.encode((self.task_start_token), add_special_tokens=False)
            ).unsqueeze(0)
        ).to(self.device_env.device)  # Adding extra dimension for batch
       # breakpoint()
        generation = self.model.text_decoder.trunk.generate(input_ids=input_ids, encoder_outputs=image_output, pad_token_id=1, max_new_tokens=512)
        return generation

    def evaluate(self, sample: dict):

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

        ocr_metrics, _ = get_ocr_metrics(
        model=self.model,
        tokenizer=self.tokenizer,
        image_input=image_input,
        text_input=text_target,
        device_env=self.device_env,
        max_recursion_length=self.max_recursion_length,
        prompt_token=self.task_start_token,
    )

        metrics["ocr_reconstruction"] = ocr_metrics
        return metrics

cfg_path =  '/fsx/dana_aubakirova/pixparse/src/pixparse/models/configs/cruller_vitL384_qk_siglip.json'
#checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoints/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoint-89.pt'
#checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoints/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoint-89.pt'
checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240426-161012-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_60/checkpoints/20240426-161012-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_60/checkpoint-49.pt'

img_path = '/fsx/dana_aubakirova/FUNSD-000000/0000971160.png'

#cfg = ModelCfg.from_json(cfg_path)
device_env = DeviceEnv()
evaluator = SmallEval(cfg_path, checkpoint_path, device_env)
evaluator.setup()
output = evaluator.prepare_image_output(img_path)
preds = evaluator.get_preds(output)

text_path = '/fsx/dana_aubakirova/FUNSD-000000/0000971160.txt'
orig_text = open(text_path, 'r').read()
orig_text = orig_text.split('\n')

#with torch.inference_mode():
   # generated_texts = evaluator.tokenizer.decode(preds) for preds in preds.tolist()
#apply wer transforms on the text

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
decoded_texts = evaluator.tokenizer.batch_decode(preds)
# Transform original and predicted texts
ocr_predictions = [re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in decoded_texts]
orig_texts = [re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in orig_text]

#cer_ocr_predictions = cer_transforms(ocr_predictions)
#cer_orig_texts = cer_transforms(orig_texts)
#wer_ocr_predictions = wer_transforms(ocr_predictions)
#wer_orig_texts = wer_transforms(orig_texts)

# Calculate WER and CER avoiding empty strings
wer_sum, cer_sum = 0, 0
valid_pairs = 0

for ref, pred in zip(orig_texts, ocr_predictions):
    if ref and pred:  # Ensure neither the reference nor prediction is empty
        wer_sum += wer(ref, pred)
        cer_sum += cer(ref, pred)
        #valid_pairs += 1

#calculate wer
wer_avg = wer_sum / len(orig_texts)
cer_avg = cer_sum / len(orig_texts)
# Avoid division by zero

print(f'WER: {wer_avg:.2f}, CER: {cer_avg:.2f}')
print(ocr_predictions)
print(orig_texts)


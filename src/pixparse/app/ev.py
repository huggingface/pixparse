import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import re
import jiwer
from typing import  Union
from pixparse.models import create_model
from pixparse.models.config import ModelCfg
from pixparse.tokenizers import create_tokenizer, TokenizerCfg
from pixparse.framework import DeviceEnv

class ImageTextDataset(Dataset):
    def __init__(self, folder_path, img_mean, img_std):
        self.folder_path = folder_path
        self.pairs = [
            (os.path.join(folder_path, f'{name}'), os.path.join(folder_path, f'{name[:-4]}.txt'))
            for name in os.listdir(folder_path) if name.endswith('.png')
        ]
        self.img_mean = img_mean
        self.img_std = img_std 
        self.transform = transforms.Compose([
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
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        with open(txt_path, 'r') as file:
            text = file.read().strip()
        return image, text

class SmallEval:
    def __init__(self, cfg: Union[str, ModelCfg], checkpoint_path: str, device_env: DeviceEnv):
        self.cfg = ModelCfg.load(cfg) if isinstance(cfg, str) else cfg
        self.checkpoint_path = checkpoint_path
        self.device_env = device_env
        cfg_tokenizer = TokenizerCfg(name=self.cfg.text_decoder.name)
        self.tokenizer = create_tokenizer(cfg_tokenizer)
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
        self.model = create_model(self.cfg, new_vocab_size=len(self.tokenizer), pretrained=checkpoint_path)
        self.model.eval().to(device_env.device)
        self.img_mean = self.model.image_encoder.trunk.pretrained_cfg["mean"]
        self.img_std = self.model.image_encoder.trunk.pretrained_cfg["std"]
        self.img_mean = (
        sum(self.img_mean) / len(self.img_mean)
    #if taskcfg.model.image_encoder.image_fmt == "L"
    #else img_mean 
        )
        self.img_std = (
            sum(self.img_std) / len(self.img_std)
            #if taskcfg.model.image_encoder.image_fmt == "L"
            #else img_std
        )

    def prepare_image_output(self, image):
        return self.model.image_encoder(image.to(self.device_env.device))

    def get_preds(self, image_output):
        input_ids = torch.tensor(self.tokenizer.encode("<s_pretrain>", add_special_tokens=False)).unsqueeze(0).to(self.device_env.device)
        return self.model.text_decoder.trunk.generate(input_ids=input_ids, encoder_outputs=image_output, pad_token_id=1, max_new_tokens=512)

    def evaluate(self, folder_path):
        dataset = ImageTextDataset(folder_path, self.img_mean, self.img_std)
        wer_sum, cer_sum, count = 0, 0, 0
        cer_transform = jiwer.transforms.Compose([
            jiwer.transforms.RemoveMultipleSpaces(),
            jiwer.transforms.Strip(),
            jiwer.transforms.ReduceToListOfListOfChars(),
        ])
        wer_transform = jiwer.transforms.Compose([
            jiwer.transforms.RemoveMultipleSpaces(),
            jiwer.transforms.Strip(),
            jiwer.transforms.ReduceToListOfListOfWords(),
        ])
        
        for image, orig_text in DataLoader(dataset, batch_size=1):
            with torch.inference_mode():
                image_output = self.prepare_image_output(image[0].unsqueeze(0))
            preds = self.get_preds(image_output)
            decoded_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            #decoded_text = ' '.join(decoded_texts)

            #orig_text_transformed = wer_transform(orig_text[0])
            #decoded_text_transformed = wer_transform(decoded_text)
            ocr_predictions = [re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in decoded_texts]
            orig_texts = [re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in orig_text]
            wer_score = jiwer.wer(orig_texts, ocr_predictions)
            cer_score = jiwer.cer(orig_texts, ocr_predictions)
            #orig_text_transformed = cer_transform(orig_text[0])
            #decoded_text_transformed = cer_transform(decoded_text)

            wer_sum += wer_score
            cer_sum += cer_score
            count += 1

        if count > 0:
            wer_avg = wer_sum / count
            cer_avg = cer_sum / count
            print(f'Average WER: {wer_avg:.2f}, Average CER: {cer_avg:.2f}, Count: {count}')
        else:
            print("No valid pairs found for evaluation.")

# Example usage:
cfg_path =  '/fsx/dana_aubakirova/pixparse/src/pixparse/models/configs/cruller_vitL384_qk_siglip.json'
#checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240422-142136-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_10/checkpoints/20240422-142136-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_10/checkpoint-9.pt'
#checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoints/20240422-140801-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoint-29.pt'
#checkpoint_path = '/fsx/dana_aubakirova/pixparse-exps/20240425-124126-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_30/checkpoints/20240425-124126-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_30/checkpoint-29.pt'
#chkpt1 = '/fsx/dana_aubakirova/pixparse-exps/20240427-160315-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoints/20240427-160315-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoint-9.pt'
#chkpt2 = '/fsx/dana_aubakirova/pixparse-exps/20240427-162933-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoints/20240427-162933-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_90/checkpoint-89.pt'
chkpt3 = '/fsx/dana_aubakirova/pixparse-exps/20240426-161012-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_60/checkpoints/20240426-161012-task_cruller_pretrain-model_cruller_vitL384_qk_siglip-lr_3.0e-05-b_36-intervals_60/checkpoint-49.pt'
folder_path = '/fsx/dana_aubakirova/FUNSD-000000/'
device_env = DeviceEnv()

evaluator = SmallEval(cfg_path, chkpt3, device_env)
evaluator.evaluate(folder_path)

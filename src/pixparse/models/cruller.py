import torch.nn as nn
import torch

import PIL

from typing import Optional

from .config import ModelCfg
from .image_encoder_timm import ImageEncoderTimm
from .text_decoder_hf import TextDecoderHf

import re

from transformers.file_utils import ModelOutput

class Cruller(nn.Module):
    def __init__(self, cfg: ModelCfg, tokenizer): #FIXME we need to pass along something like a TokenizerCfg
        super().__init__()
        self.image_encoder = ImageEncoderTimm(cfg.image_encoder)
        self.text_decoder = TextDecoderHf(cfg.text_decoder, tokenizer)

    def forward(self, image_input, text_input):
        encoder_output = self.image_encoder(image_input)
        decoder_output = self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            return_dict=True,
        )
        return decoder_output
    
    def inference(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        try:
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.cfg.device)
        except:
            # half is not compatible in cpu implementation.
            pass

        if prompt_tensors is None:
            prompt_tensors = self.text_decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.cfg.device)

        last_hidden_state = self.image_encoder(image_tensors)
        if self.cfg.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.text_decoder.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.text_decoder.tokenizer.pad_token_id,
            eos_token_id=self.text_decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.text_decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {"predictions": list()}
        for seq in self.text_decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.text_decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

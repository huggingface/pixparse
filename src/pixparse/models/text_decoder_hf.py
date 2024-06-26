from typing import Optional

import torch
import transformers
from torch import nn as nn

from pixparse.models.config import TextDecoderCfg


def create_text_decoder(cfg: TextDecoderCfg) -> transformers.BartForCausalLM:  # FIXME for type hints
    assert cfg.name

    config = transformers.AutoConfig.from_pretrained(cfg.name)
    config.add_cross_attention = True
    if False:  # FIXME this were set in Donut but missed in first pass, should compare
        config.is_encoder_decoder = False
        config.scale_embedding = True
        config.add_final_layer_norm = True
    if cfg.num_decoder_layers is not None:
        config.decoder_layers = cfg.num_decoder_layers
    if cfg.max_length is not None:
        config.max_position_embeddings = cfg.max_length
    #config.vocab_size =   # FIXME set vocab size here or rely on model resize when tokens added?

    if cfg.pretrained:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.name,
            config=config,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_config(
            config,
        )
    # TODO Following is the donut hack. Unused without generate().
    # model.model.decoder.embed_tokens.padding_idx = cfg.pad_token_id

    return model


class TextDecoderHf(nn.Module):

    def __init__(self, cfg: TextDecoderCfg):
        super().__init__()
        self.trunk = create_text_decoder(cfg)
        self.prepare_inputs_for_generation = self.prepare_inputs_for_inference

    def prepare_inputs_for_inference(
            self,
            input_ids: torch.Tensor,
            encoder_outputs: torch.Tensor,
            pad_token_id: int,
            past_key_values=None,
            past=None,
            use_cache: bool = None,
            attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        # for compatibility with transformers==4.11.x
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs #.last_hidden_state, #FIXME for timm ViT encoder there is no last hidden state
        }
        return output

    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            use_cache: bool = None,
            output_attentions: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[torch.Tensor] = None,
            return_dict: bool = None,
    ):
        # FIXME is this always going to be a direct pass through or will some tasks/models
        # need extra logic before/after trunk.forward()
        output = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output

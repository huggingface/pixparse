from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn
from transformers.models.bart import BartConfig
from transformers.models.bart.modeling_bart import BartAttention, BartDecoderLayer
from transformers.activations import ACT2FN


class PixParseBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if qk_norm:
            self.q_norm = nn.LayerNorm(embed_dim)
            self.k_norm = nn.LayerNorm(embed_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # FIXME _shape name is ambiguous
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_norm(self.q_proj(hidden_states)), tgt_len, bsz)

        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            kv_input = key_value_states if is_cross_attention else hidden_states
            key_states = self._shape(self.k_norm(self.k_proj(kv_input)), -1, bsz)
            value_states = self._shape(self.v_proj(kv_input), -1, bsz)
            if past_key_value is not None:
                assert not is_cross_attention
                # reuse k,v
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        src_len = key_states.size(-2)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )

        if use_fused_attn() and not layer_head_mask:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0,
            )
        else:
            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            query_states = query_states.view(*proj_shape) * self.scaling
            key_states = key_states.reshape(*proj_shape)
            value_states = value_states.reshape(*proj_shape)

            if attention_mask is not None:
                if attention_mask is not None:
                    attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1).view(-1, tgt_len, src_len)
                attn_weights = torch.baddbmm(attention_mask, query_states, key_states.transpose(1, 2))
            else:
                attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            attn_weights = torch.softmax(attn_weights, dim=-1)

            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_probs, value_states)

            if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


class PixParseBartDecoderLayer(nn.Module):
    def __init__(
            self,
            config: BartConfig,
            qk_norm_cross: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = PixParseBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = PixParseBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            qk_norm=qk_norm_cross,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def convert_bart_pp(module: nn.Module, config: BartConfig, qk_norm_cross=False, qk_norm=False):
    """
    Utility to convert HF Bart to pixparse Bart.
    """
    module_output = module
    if isinstance(module, BartDecoderLayer):
        module_output = PixParseBartDecoderLayer(
            config=config,
            qk_norm_cross=qk_norm_cross,
        )

        def _attn(mo: PixParseBartAttention, m: BartAttention):
            for n in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                getattr(mo, n).weight = getattr(m, n).weight
                getattr(mo, n).bias = getattr(m, n).bias

        with torch.no_grad():
            _attn(module_output.self_attn, module.self_attn)
            _attn(module_output.encoder_attn, module.encoder_attn)
            for n in ('self_attn_layer_norm', 'encoder_attn_layer_norm', 'fc1', 'fc2', 'final_layer_norm'):
                getattr(module_output, n).weight = getattr(module, n).weight
                getattr(module_output, n).bias = getattr(module, n).bias
    else:
        for name, child in module.named_children():
            module_output.add_module(
                name,
                convert_bart_pp(child, config=config, qk_norm_cross=qk_norm_cross, qk_norm=qk_norm)
            )
    del module
    return module_output


def _resize_embeddings(weight: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Helper method to resize embeddings tensor.
    """
    if weight.size(0) > max_length:
        # Truncate the positional embeddings if necessary
        resized_weight = weight[:max_length, :]
    else:
        # Interpolate the positional embeddings if necessary
        resized_weight = F.interpolate(
            weight.unsqueeze(0).permute(0, 2, 1),
            size=max_length,
            mode='linear',
            align_corners=False
        ).squeeze(0).permute(1, 0)
    return resized_weight


def resize_positional_embeddings(module: nn.Module, max_length: int) -> nn.Module:
    """
    Resizes the `decoder.embed_positions.weight` of bart. 
    # FIXME model summary weights are not updated with the current method, so the print is unreliable.
    """
    if max_length != module.config.max_position_embeddings:
        module.model.decoder.embed_positions.weight = torch.nn.Parameter(
            _resize_embeddings(
                module.model.decoder.embed_positions.weight,
                max_length
                + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
            )
        )

        module.config.max_position_embeddings = max_length

    return module


# When generating with fsdp, make sure to use the forward function from the unwrapped model
# 
import os
from typing import Union, Optional
from enum import Enum

from .config import ModelCfg, get_model_config
from .cruller import Cruller
from ..utils import load_checkpoint, check_exists
import torch


class ModelType(Enum):
    CRULLER = Cruller
    # OTHER_PASTRY = OtherPastry


def get_model(model_cfg: Union[str, ModelCfg]):
    """
    Retrieves the model instance based on the provided configuration.

    Args:
        model_cfg (Union[str, ModelCfg]): The model configuration, which can be a string or a ModelCfg instance.

    Returns:
        An instance of the specified model.

    # TODO add support of HF-based models here.
    """
    try:
        model_type = ModelType[model_cfg.type.upper()].value
    except KeyError:
        valid_types = [model.name for model in ModelType]
        raise ValueError(f"Invalid model type, got {model_cfg.type}, supported models are  {valid_types}")
    return model_type(model_cfg)

def resize_model_embeddings(model, num_new_tokens: int):
    # TODO make this method call the right resize depending on model structure. 
    model.text_decoder.trunk.resize_token_embeddings(num_new_tokens)


def create_model(
        model_name_or_cfg: Union[str, ModelCfg],
        pretrained: Optional[str] = '',
        num_new_tokens: Optional[int] = 0,
):
    """
    Creates and initializes a model based on the given configuration and optional pretrained state.

    Args:
        model_name_or_cfg (Union[str, ModelCfg]): The name of the model or its configuration.
        pretrained (str, optional): Path to the pretrained model's state. Defaults to ''.
        num_new_tokens (int, optional): Number of new tokens to add to the model's embeddings. Defaults to 0.

    Returns:
        The initialized model.
    """
    if isinstance(model_name_or_cfg, str):
        model_cfg = get_model_config(model_name_or_cfg)
    else:
        assert isinstance(model_name_or_cfg, ModelCfg)
        model_cfg = model_name_or_cfg

    model = get_model(model_cfg)

    if pretrained:
        if check_exists(pretrained):
            # FIXME replace with a load fn that can adapt resolutions, input channels, heads, etc
            checkpoint = load_checkpoint(pretrained)
            model.load_state_dict(checkpoint['model'])
        else:
            assert False, 'other pretrained modes WIP'

    if num_new_tokens > 0:
        resize_model_embeddings(model, num_new_tokens)

    return model


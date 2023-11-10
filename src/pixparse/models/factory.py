import os
from typing import Union

from .config import ModelCfg, get_model_config
from .cruller import Cruller

import torch

def create_model(
        model_name_or_cfg: Union[str, ModelCfg],
        pretrained: str = '',
):
    if isinstance(model_name_or_cfg, str):
        model_cfg = get_model_config(model_name_or_cfg)
    else:
        assert isinstance(model_name_or_cfg, ModelCfg)
        model_cfg = model_name_or_cfg

    # FIXME support HF hub based models like Donut, Nougat, Pix2Struct here or elsewhere?
    assert model_cfg.type == 'cruller'
    model = Cruller(model_cfg)

    if pretrained:
        if os.path.isfile(pretrained):
            # FIXME replace with a load fn that can adapt resolutions, input channels, heads, etc
            # Should also handle whether or not to load optimizer, scheduler state? or just model? - molbap
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'])
        else:
            assert False, 'other pretrained modes WIP'

    return model

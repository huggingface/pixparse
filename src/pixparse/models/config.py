import copy
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import ClassVar, Optional, Tuple

from simple_parsing.helpers import Serializable

from pixparse.utils.name_utils import _natural_key, clean_name

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"configs/"]
_MODEL_CONFIGS = {}  # model_name: config


_logger = logging.getLogger(__name__)


@dataclass
class ImageEncoderCfg(Serializable):
    name: str = 'vit_base_patch16_224'
    image_fmt: str = 'L'
    image_size: Optional[Tuple[int, int]] = (576, 448)
    needs_image_size: bool = True  # model needs static image size on creation (vit, swin, etc)
    pretrained: bool = True


@dataclass
class TextDecoderCfg(Serializable):
    name: str = 'facebook/bart-base'
    pretrained: bool = True
    num_decoder_layers: Optional[int] = 4
    max_length: Optional[int] = 1024
    pad_token_id: Optional[int] = None  # FIXME move this to TokenizerCfg?


@dataclass
class ModelCfg(Serializable):
    type: str = 'cruller'
    image_encoder: ImageEncoderCfg = field(default_factory=ImageEncoderCfg)
    text_decoder: TextDecoderCfg = field(default_factory=TextDecoderCfg)


# FIXME I'm not super happy about the cfg / args arrangement. Improves some issues, creates others -rwightman
@dataclass
class ModelArgs:
    # specify a preset model configuration to load
    name: str = ""

    # common overrides
    image_fmt: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    text_max_length: Optional[int] = None

    # final, loaded & merged config
    cfg: ModelCfg = field(init=False)  # FIXME better to move this into task directly?

    def __post_init__(self):
        model_cfg = get_model_config(self.name)
        if model_cfg is None:
            _logger.error(
                f"Model config for {self.name} was not found."
            )
            raise RuntimeError(f"Model config for {self.name} was not found.")

        if self.image_size is not None:
            replace(model_cfg.image_encoder, image_size=self.image_size)
        if self.image_fmt is not None:
            replace(model_cfg.image_encoder, image_fmt=self.image_fmt)
        if self.text_max_length is not None:
            replace(model_cfg.text_decoder, max_length=self.text_max_length)

        self.cfg = model_cfg


def _scan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        model_cfg = ModelCfg.load(cf)
        _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_scan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    model_name = clean_name(model_name)
    cfg = _MODEL_CONFIGS.get(model_name, None)
    return copy.deepcopy(cfg)

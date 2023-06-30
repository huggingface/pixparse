import copy
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

from simple_parsing.helpers import Serializable


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"configs/"]
_MODEL_CONFIGS = {}  # model_name: config


@dataclass
class ImageEncoderCfg(Serializable):
    name: str = 'vit_base_patch16_224'
    image_fmt: str = 'L'
    image_size: Optional[Tuple[int, int]] = (576, 448)
    pretrained: bool = True


@dataclass
class TextDecoderCfg(Serializable):
    name: str = 'facebook/bart-base'
    pretrained: bool = True
    num_decoder_layers: Optional[int] = 4
    max_length: Optional[int] = 1024
    pad_token_id: Optional[int] = None


@dataclass
class ModelCfg(Serializable):
    image_encoder: ImageEncoderCfg = field(default_factory=ImageEncoderCfg)
    text_decoder: TextDecoderCfg = field(default_factory=TextDecoderCfg)


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


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


def clean_model_name(name):
    name = name.replace('/', '_')
    name = name.replace('-', '_')
    return name


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    model_name = clean_model_name(model_name)
    cfg = _MODEL_CONFIGS.get(model_name, None)
    return copy.deepcopy(cfg)

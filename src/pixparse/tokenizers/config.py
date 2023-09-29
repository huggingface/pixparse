import copy
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

from simple_parsing.helpers import Serializable

from pixparse.utils.name_utils import _natural_key, clean_name

_TOKENIZER_CONFIG_PATHS = [Path(__file__).parent / f"configs/"]
_TOKENIZER_CONFIGS = {}  # model_name: config


@dataclass
class TokenizerCfg(Serializable):
    name: str = 'facebook/bart-large'
    pretrained: bool = True


def _scan_tokenizer_configs():
    global _TOKENIZER_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _TOKENIZER_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        tokenizer_cfg = TokenizerCfg.load(cf)
        _TOKENIZER_CONFIGS[cf.stem] = tokenizer_cfg

    _TOKENIZER_CONFIGS = {k: v for k, v in sorted(_TOKENIZER_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_scan_tokenizer_configs()  # initial populate of model config registry


def list_tokenizers():
    """ enumerate available model architectures based on config files """
    return list(_TOKENIZER_CONFIGS.keys())


def get_tokenizer_config(tokenizer_name):
    tokenizer_name = clean_name(tokenizer_name)
    cfg = _TOKENIZER_CONFIGS.get(tokenizer_name, None)
    return copy.deepcopy(cfg)

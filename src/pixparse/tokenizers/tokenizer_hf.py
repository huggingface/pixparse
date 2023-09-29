from pixparse.tokenizers.config import TokenizerCfg
from transformers import AutoTokenizer


def create_tokenizer(cfg: TokenizerCfg):
    assert cfg.name
    extra_kwargs = {}  # FIXME do we want to pass additional_special_tokens here? they are task-specific
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name,
        **extra_kwargs
        )
    return tokenizer

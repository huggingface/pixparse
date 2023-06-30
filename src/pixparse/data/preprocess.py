import logging
from typing import Callable

import torch

_logger = logging.getLogger(__name__)


def preprocess_text_anno(
        anno,
        tokenizer: Callable,
        max_position_embeddings: int,
        task_start_token: str,
        prompt_end_token: str,
        ignore_id: int = -100,
        generator=None,
):
    """
    Simpler data preprocessing for raw-text data.
    """
    text = task_start_token + anno + tokenizer.eos_token

    tokenizer_fn = lambda x: tokenizer(
        x,
        add_special_tokens=False,
        return_tensors='pt',
        max_length=max_position_embeddings,
        padding='max_length',
        truncation=True).input_ids[0]

    text = tokenizer_fn(text)

    target = text.clone()
    # model doesn't need to predict pad token
    target[target == tokenizer.pad_token_id] = ignore_id
    # model doesn't need to predict prompt (for VQA)
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
    target[:torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id

    return dict(text=[text], target=[target])


def preprocess_ocr_anno(
        anno,
        tokenizer: Callable,
        max_position_embeddings: int,
        task_start_token: str,
        prompt_end_token: str,
        ignore_id: int = -100,
        generator=None,
):
    # FIXME complete and update this fn to match our OCR annotation format
    if isinstance(anno, list):
        # FIXME this was an intermediate annotation form, should not exist anymore
        _logger.warning("Old [id, {}] annotation form found, correcting...")
        anno = anno[1]

    num_pages = len(anno['pages'])
    if not num_pages:
        raise RuntimeError("Empty annotation. Skipping...")

    tokenizer_fn = lambda x: tokenizer(
        x,
        add_special_tokens=False,
        return_tensors='pt',
        max_length=max_position_embeddings,
        padding='max_length',
        truncation=True).input_ids[0]
    pad_token_id = tokenizer.pad_token_id
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)

    # FIXME for initial behaviour we will randomly sample one of N pages
    # TODO determine if we want to train in multi-page mode, use another sampling strategy?
    page_indices = [generator.randint(0, num_pages - 1)]
    # page_indices = range(num_pages)
    text_pages = []
    target_pages = []
    for i in page_indices:
        # FIXME treating pages separately, this best approach or tokenize w/ page-break?
        anno_page = anno['pages'][i]
        if not anno_page['text']:
            raise RuntimeError("No text on page, skipping...")

        # FIXME see self.donut_model.json2token, task specific json tokenization for
        #  the non-pretrain tasks varies w/ differing special tokens and prompts
        text = '\n'.join(anno_page['text'])
        orig_text = text
        text = task_start_token + text + tokenizer.eos_token
        text = tokenizer_fn(text)

        target = text.clone()
        # model doesn't need to predict pad token
        target[target == pad_token_id] = ignore_id
        # model doesn't need to predict prompt (e.g. VQA)
        target[:torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id

        text_pages.append(text)
        target_pages.append(target)

    return dict(text=text_pages, target=target_pages), dict(page_indices=page_indices, num_pages=num_pages, orig_text=orig_text)

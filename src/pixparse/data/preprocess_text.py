import logging
from typing import Callable

import torch

_logger = logging.getLogger(__name__)


def text_input_to_target(text_input, tokenizer, prompt_end_token, ignore_id=-100):
    target = text_input.clone()
    # model doesn't need to predict pad token
    target[target == tokenizer.pad_token_id] = ignore_id
    # model doesn't need to predict prompt (for VQA)
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
    target[: torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id
    return target


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
        return_tensors="pt",
        max_length=max_position_embeddings,
        padding="max_length",
        truncation=True,
    ).input_ids[0]

    text = tokenizer_fn(text)

    target = text.clone()
    # model doesn't need to predict pad token
    target[target == tokenizer.pad_token_id] = ignore_id
    # model doesn't need to predict prompt (for VQA)
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
    target[: torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id

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

    num_pages = len(anno["pages"])
    if not num_pages:
        raise RuntimeError("Empty annotation. Skipping...")

    tokenizer_fn = lambda x: tokenizer(
        x,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_position_embeddings,
        padding="max_length",
        truncation=True,
    ).input_ids[0]
    pad_token_id = tokenizer.pad_token_id
    prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)

    # FIXME for initial behaviour we will randomly sample one of N pages
    # TODO determine if we want to train in multi-page mode, use another sampling strategy?
    current_index = generator.randint(0, num_pages - 1)
    if not anno["pages"][current_index]["text"]:
        current_index = get_next_valid_page_index(current_index, num_pages, anno)

    page_indices = []
    text_pages = []
    target_pages = []
    n_wanted_pages = min(
        1, num_pages
    )  # TODO increase that number for multipage processing
    while len(text_pages) < n_wanted_pages:
        # FIXME treating pages separately, this best approach or tokenize w/ page-break?
        anno_page = anno["pages"][current_index]
        if not anno_page["text"]:
            raise RuntimeError("No text on page, skipping...")
        # FIXME see self.donut_model.json2token, task specific json tokenization for
        #  the non-pretrain tasks varies w/ differing special tokens and prompts
        text = "\n".join(anno_page["text"])
        orig_text = text
        text = task_start_token + text + tokenizer.eos_token
        text = tokenizer_fn(text)

        target = text.clone()
        # model doesn't need to predict pad token
        target[target == pad_token_id] = ignore_id
        # model doesn't need to predict prompt (e.g. VQA)
        target[: torch.nonzero(target == prompt_end_token_id).sum() + 1] = ignore_id

        text_pages.append(text)
        target_pages.append(target)
        page_indices.append(current_index)

        current_index = get_next_valid_page_index(current_index, num_pages, anno)

    info = dict(page_indices=page_indices, num_pages=num_pages, orig_text=orig_text)
    return dict(text=text_pages, target=target_pages), info


def get_next_valid_page_index(current_index: int, num_pages: int, anno: dict, retries: int = 10):
    """
    Get the index of the next valid page which contains text. If it doesn't find any non empty page
    after 'retries' attempts, it raises a RuntimeError.

    Parameters:
    current_index (int): Current page index.
    num_pages (int): Total number of pages.
    anno (dict): The annotation dictionary which contains the 'pages'.
    retries (int): Number of maximum retries for a given document.

    Returns:
    int: The index of the next non empty page.
    """
    for _ in range(retries):
        current_index = (
            current_index + 1
        ) % num_pages  # Get the next index, wrap around to 0 if it exceeds num_pages (in case of random init)
        anno_page = anno["pages"][current_index]
        if anno_page["text"]:
            return current_index
    raise RuntimeError(f"No non-empty page found after {retries} attempts")

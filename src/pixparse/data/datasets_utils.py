from ast import literal_eval

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from pixparse.utils.json_utils import json2token

"""
This util file regroups necessary preprocessing steps 
that have do be done before scaling up a model training.
In particular, new tokens have to be found and added to the tokenizer
through a first pass of the dataset.

In particular for json processing of datasets, each key becomes a
new special token, and the tokenizer vocab needs to be resized on the fly. 
"""


class SafeDataset:
    """
    This is a Dataset wrapped by a try/except in the __getitem__ in case
    the hf datasets used have errors/corrupt data.
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        try:
            item = self.original_dataset[idx]
            return item
        except Exception as e:
            return None


def get_additional_tokens_from_dataset(all_special_tokens:list, dataset=None, dataset_id:str="naver-clova-ix/cord-v2")->list:
    """
    This util is made to run a first pass for CORD
    with an instantiated tokenizer.
    the additional tokens are returned as a list and can then be
    added to your tokenizer and saved
    to disk.

    Usage:
    # Instantiate tokenizer for your task
    taskcfg = TaskCrullerPretrainCfg(model_name="cruller_base")
    tokenizer = TokenizerHF(taskcfg.tokenizer)
    all_special_tokens = tokenizer.trunk.all_special_tokens

    new_special_tokens = get_additional_tokens_from_dataset(all_special_tokens, dataset_id="naver-clova-ix/cord-v2")

    # Now you can add the tokens
    newly_added_num = tokenizer.trunk.add_special_tokens(
        {"additional_special_tokens": sorted(set(new_special_tokens))}
    )

    # You can resize the embeddings of your text decoder accordingly

    if newly_added_num > 0:
        model.text_decoder.trunk.resize_token_embeddings(
            len(tokenizer.trunk)
        )

    # now your tokenizer will parse correctly the dataset.
    """
    if dataset_id == "naver-clova-ix/cord-v2":

        def collate_fn(batch):
            """
            basic collator for PIL images, as returned by rvlcdip dataloader (among others)
            """
            text_inputs = [
                literal_eval(item["ground_truth"])["gt_parse"] for item in batch
            ]
            return {"label": text_inputs}

        cord = load_dataset(dataset_id)
        loader = DataLoader(cord["train"], batch_size=32, collate_fn=collate_fn)


        new_special_tokens = []
        for i, batch in enumerate(loader):
            for text in batch['label']:
                _, batch_special_tokens = json2token(text, all_special_tokens)
                new_special_tokens += batch_special_tokens
                new_special_tokens = list(set(new_special_tokens))
    return new_special_tokens
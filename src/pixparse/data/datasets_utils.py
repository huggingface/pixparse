import json
import os
from ast import literal_eval

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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

class CustomVQADataset(Dataset):
    """
    Custom implementation of the SinglePageDocVQA dataset.
    len(train_images), len(test_images), len(val_images)
    is (10194, 1287, 1286). Each image has one or more questions and answers attached, or solely questions for the test set.
    However, in terms of questions, there are 
    5188 in the test set. 
    5349 in the val set.
    """
    def __init__(self, root_dir, split, transform=None):
        self.extra_tokens = ['<s_answer>', '</s_answer>', '</s_question>', '<s_question>']
        self.root_dir = root_dir
        self.split = split
        assert split in ["train", "test", "val"], "split is not train, test or val."
        if split == "test" or split == "val":
            json_path = os.path.join(root_dir, split, f"{split}_v1.0.json")
        else:
            json_path = os.path.join(root_dir, split, f"processed_{split}_v1.0.json")
        assert os.path.isdir(self.root_dir), f"Can't find {root_dir}. Make sure you have DocVQA files locally."
        assert os.path.isfile(json_path), f"{json_path} not found. Make sure you have the processed dataset."
        self.img_dir = os.path.join(root_dir, split)
        
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)
        self.all_images = list(self.data_dict.keys())
        self.transform = transform
        if split == "train":
            self.train_data = []
            for image_id, qas in self.data_dict.items():
                for qa in qas:
                    self.train_data.append([image_id, qa]) 
    
    def __len__(self):
        if self.split == "test" or self.split == "val":
            return len(self.data_dict['data'])
        else:
            return len(self.train_data)
    
    def __getitem__(self, index):
        if self.split == "test":
            entry = self.data_dict['data'][index]
            labels = "<s_question>" + entry['question'] + "</s_question>"
            img_path = os.path.join(self.img_dir, entry['image'])
            question_id = entry['questionId']
            image_id = entry["image"]
        if self.split == "val":
            entry = self.data_dict['data'][index]
            labels = {"question": entry['question'], "answers": entry['answers']}
            img_path = os.path.join(self.img_dir, entry['image'])
            question_id = entry['questionId']
            image_id = entry["image"]
        else:
            image_id, labels = self.train_data[index]
            img_path = os.path.join(self.img_dir, image_id)
            question_id = -1 # Not parsed from original dataset.
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "labels": labels, "image_id": image_id, "question_id": question_id}

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
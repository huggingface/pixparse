from datasets import load_dataset

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
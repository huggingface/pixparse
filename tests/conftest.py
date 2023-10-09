import pytest
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

from pixparse.task.task_cruller_finetune_CORD import CollateCORD
from pixparse.task.task_cruller_finetune_docvqa import CollateDocVQA
from pixparse.task.task_cruller_finetune_RVLCDIP import CollateRVLCDIP


@pytest.fixture
def docvqa_train_sample():
    return {
        'image': Image.new('L', (576, 448)),
        'labels': [
            """<s_question> What is the ‘actual’ value per 1000, during the year 1975?</s_question>
            <s_answer>0.28</s_answer>""",
            """<s_question> What is the ‘actual’ value per 1000, during the year 1975?</s_question>
            <s_answer>.28</s_answer>"""
        ],
        'image_id': 'documents/pybv0228_81.png',
        'question_id': 49153
    }


@pytest.fixture
def cord_train_sample():
    return {
        'image': Image.new('L', (1108, 1478)),
        'ground_truth': """{
                    "gt_parse": {
                        "menu": [
                        {"nm": "Item1", "cnt": "2", "price": "10,000"},
                        {"nm": "Item2", "cnt": "1", "price": "5,000"}
                        ],
                        "sub_total": {
                        "subtotal_price": "25,000",
                        "discount_price": "5,000",
                        "service_price": "1,000",
                        "tax_price": "2,000"
                        },
                        "total": {"total_price": "23,000"}
                    },
                    "meta": {
                        "version": "2.0.0",
                        "split": "train",
                        "image_id": 3,
                        "image_size": {"width": 1108, "height": 1478}
                    },
                    "valid_line": [
                        {
                        "words": [
                            {
                            "quad": {"x1": 10, "y1": 10, "x2": 20, "y2": 10, "x3": 20, "y3": 20, "x4": 10, "y4": 20},
                            "is_key": 0,
                            "row_id": 1,
                            "text": "Item1"
                            }
                        ],
                        "category": "menu.nm",
                        "group_id": 1,
                        "sub_group_id": 0
                        },
                        {
                        "words": [
                            {
                            "quad": {"x1": 10, "y1": 30, "x2": 20, "y2": 30, "x3": 20, "y3": 40, "x4": 10, "y4": 40},
                            "is_key": 0,
                            "row_id": 1,
                            "text": "2"
                            }
                        ],
                        "category": "menu.cnt",
                        "group_id": 1,
                        "sub_group_id": 0
                        }
                    ],
                    "roi": {"x1": 10, "y1": 10, "x2": 100, "y2": 10, "x3": 100, "y3": 100, "x4": 10, "y4": 100},
                    "repeating_symbol": [],
                    "dontcare": []
                    }"""

    }


@pytest.fixture
def rvlcdip_train_sample():
    return {
        'image': Image.new('L', (576, 448)),
        'label': 0
    }


@pytest.fixture
def docvqa_train_batch(docvqa_train_sample):
    return [docvqa_train_sample] * 4


@pytest.fixture
def cord_train_batch(cord_train_sample):
    return [cord_train_sample] * 4


@pytest.fixture
def rvlcdip_train_batch(rvlcdip_train_sample):
    return [rvlcdip_train_sample] * 4


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained('facebook/bart-base')


@pytest.fixture
def image_preprocess():
    return transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def docvqa_collator(tokenizer, image_preprocess):
    tokenizer.add_special_tokens({"additional_special_tokens": sorted({'<s_docvqa>', '</s_docvqa>'})})
    return CollateDocVQA(
        tokenizer=tokenizer,
        image_preprocess=image_preprocess,
        start_token="<s_docvqa>",
        max_length=512,
        end_token=tokenizer.eos_token
    )


@pytest.fixture
def cord_collator(tokenizer, image_preprocess):
    tokenizer.add_special_tokens({"additional_special_tokens": sorted({'<s_cord>', '</s_cord>'})})
    return CollateCORD(
        tokenizer=tokenizer,
        image_preprocess=image_preprocess,
        start_token="<s_cord>",
        max_length=512,
    )


@pytest.fixture
def rvlcdip_collator(tokenizer, image_preprocess):
    label_int2str = {
        0: "letter",
        1: "form",
        2: "email",
        3: "handwritten",
        4: "advertisement",
        5: "scientific_report",
        6: "scientific_publication",
        7: "specification",
        8: "file_folder",
        9: "news_article",
        10: "budget",
        11: "invoice",
        12: "presentation",
        13: "questionnaire",
        14: "resume",
        15: "memo",
    }
    tokenizer.add_special_tokens({"additional_special_tokens": sorted({'<s_rvlcdip>', '</s_rvlcdip>', '<letter/>'})})
    return CollateRVLCDIP(
        tokenizer=tokenizer,
        image_preprocess=image_preprocess,
        start_token="<s_rvlcdip>",
        max_length=6,
        label_int2str=label_int2str
    )

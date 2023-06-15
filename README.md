# Pixel Parsing (`pixparse`)

## Introduction

An open reproduction of OCR-free end-to-end document understanding models with open data.

Broadly focused on these model types:
* image encoder + text decoder w/ pixels and text tokens as input (as per Donut)
* image encoder + text decoder w/ only pixels as input (as per Pix2Struct)
* image encoder + text encoder-decoder w/ pixels and text tokens as input (as per PaLI/PaLI-X)

The training objectives and pretraining datasets will also be inspired by the associated papers above, but will mix and match. For example, we may train a Donut or PaLI-X style model with a Pix2Struct objective (masked document images w/ simplified HTML target text).

## Updates

2023-06-14
* Distributed training tested in a SLURM environment w/ 16x A100 over 2 nodes.

2023-06-12
* It performs train steps on image-text datasets (objective too hard to learn anything w/o text in image)
  * `python -m pixparse.app.train --train.source "/data/cc12m/cc12m-train-{0000..xxxx}.tar" --train.batch-size 8 --train.num-samples 10000000 --learning-rate 1e-4 --clip-grad-value 1.0 --clip-grad-mode norm --grad-accum-steps 4`
* Next step, trial image + ocr anno dataset

## Code Organization

Within `src/pixparse`:
* `app/` - CLI applications for training and evaluation
  * `app/train.py` - main training CLI entrypoint, will attempt to keep useable across tasks
  * `app/eval.py` - (TODO) main evaluation CLI entrypoint
  * `app/finetune.py` - (TBD) fine-tune is handled by train.py with different args/config or separate?
* `data/` - data loaders, image and text preprocessing
* `framework/` - lightweight train & evaluation scaffolding on top of canonical PyTorch
* `layers/` - custom nn.Modules and functions for re-usable modelling components
* `models/` - modelling code with associated factory methods and helpers
* `task/` - task wrappers for various objectives (model + loss fn + pre/post-processing + optimization nuances)
* `tokenizer/` - tokenizer helpers (push into data?)
* `utils/` - misc utils that don't have a home

## Concepts & Terminology

Some terms and concepts used in this project that may be a bit unfamiliar.

### Task
A key organization concept in this project. Package the model with its loss, pre/post-processing, and optimization setup together for a given objective.

Examples of tasks conceptually:
  * Pretraining a Donut style (image enc + text dec) model on supervised (OCR annotation) doc-text dataset
  * Pretraining a Pix2Struct style (image enc + text dec) model w/ a dataset of webpage/screenshots and structured, simplified HTML
  * Pretraining a PaLI style (image enc + text enc-dec) model w/ prefix & masked-token completion on datasets as above
  * Fine-tuning any of the above pretrained models on a possibly large variety of downstream tasks
    * Semi-structured doc parsing - receipt, invoice, business cards, etc.
    * VQA
    * Captioning
    * ... and more

With the Task concept, the data pipeline exists outside the task. Samples and targets are fed into the task via the step functions. The data pipeline is coupled to the task by passing the pre-processing functions created within the task to the data pipeline on creation.

### Interval

You'll see the term 'interval' in the code, sometimes next to epoch. It's related, but an epoch means 'one complete pass of the dataset', an interval may be an epoch, but it may not. Interval is a span of training between checkpoints, ideally meaningful enough in duration to warrant evaluating and archiving each interval checkpoint.

In OpenCLIP development the term arose when using shard sampling with replacement, were the intervals between checkpoints were determined by limitations on job durations or likelihood of crashes.

# Pixel Parsing (`pixparse`)

An open reproduction of OCR-free end-to-end document understanding models with open data.

Broadly focused on these model types:
* image encoder + text decoder w/ pixels and text tokens as input (as per Donut)
* image encoder + text decoder w/ only pixels as input (as per Pix2Struct)
* image encoder + text encoder-decoder w/ pixels and text tokens as input (as per PaLI/PaLI-X)

The training objectives and pretraining datasets will also be inspired by the associated papers above, but will mix and match. For example, we may train a Donut or PaLI-X style model with a Pix2Struct objective (masked document images w/ simplified HTML target text).

## Updates

2023-06-12
* It performs train steps on image-text datasets (objective too hard to learn anything w/o text in image)
  * `python -m pixparse.app.train --train.source "/data/cc12m/cc12m-train-{0000..xxxx}.tar" --train.batch-size 8 --train.num-samples 10000000 --learning-rate 1e-4 --clip-grad-value 1.0 --clip-grad-mode norm`
* Next step, trial image + ocr anno dataset 
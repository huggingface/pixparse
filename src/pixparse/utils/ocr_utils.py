import torch
from torch import nn

from jiwer import cer, wer
import jiwer.transforms as tr

import re
from typing import List
import csv

def get_ocr_metrics(
    model, tokenizer, image_input, text_input, device_env, max_recursion_length
):
    """
    Calculate OCR metrics.

    Args:
        model: Model with .image_encoder and .text_decoder attributes.
        tokenizer: The tokenizer used in the OCR model.
        image_input: Input images.
        text_input: Expected output text.

    Returns:
        A dictionary of OCR metrics, and a dictionary containing a single reconstructed sample.
    """
    cer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.Strip(),
            tr.ReduceToListOfListOfChars(),
        ]
    )

    wer_transforms = tr.Compose(
        [
            tr.RemoveSpecificWords("<pad>"),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    ocr_pretraining_metrics = dict()
    with torch.inference_mode():
        # with model.no_sync():
        if hasattr(model, "module"):  # for DDP inference
            model_attr_accessor = model.module
        else:
            model_attr_accessor = model
        image_encoding = model_attr_accessor.image_encoder(image_input)
        text_input[
            text_input == -100
        ] = (
            tokenizer.trunk.pad_token_id
        )  # FIXME the -100 id token is there to be ignored by cross entropy, we replace it by padding
        sequence_lengths = (text_input != tokenizer.trunk.pad_token_id).sum(dim=1)
        max_sequence_length = sequence_lengths.max().item()
        max_recursion_length = min(max_recursion_length, max_sequence_length)
        ocr_predictions = generate_ocr(
            model_attr_accessor,
            tokenizer,
            image_encoding,
            device_env,
            max_recursion_length,
        )
        decoded_texts = tokenizer.trunk.batch_decode(text_input)
        ocr_predictions = [
            re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in ocr_predictions
        ]
        decoded_texts = [
            re.sub(r"<.*?>", "", re.sub("\n", " ", text)) for text in decoded_texts
        ]

        ocr_predictions = [text[0:len(reftext)] for text, reftext in zip(ocr_predictions, decoded_texts)]

        ocr_pretraining_metrics = get_cer_wer_metrics(
            cer_transforms,
            wer_transforms,
            ocr_pretraining_metrics,
            ocr_predictions,
            decoded_texts,
        )
        reconstructed_sample = {
            "image": image_input[0],
            "original_text": decoded_texts[0],
            "reconstructed_text": ocr_predictions[0],
        }
    return ocr_pretraining_metrics, reconstructed_sample


def get_cer_wer_metrics(
    cer_transforms,
    wer_transforms,
    ocr_pretraining_metrics,
    ocr_predictions,
    decoded_texts,
):
    wer_output = wer(
        reference=decoded_texts,
        hypothesis=ocr_predictions,
        reference_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )
    ocr_pretraining_metrics["wer"] = wer_output
    cer_output = cer(
        reference=decoded_texts,
        hypothesis=ocr_predictions,
        reference_transform=cer_transforms,
        hypothesis_transform=cer_transforms,
    )
    ocr_pretraining_metrics["cer"] = cer_output
    return ocr_pretraining_metrics


def generate_ocr(
    model,
    tokenizer,
    encoder_outputs: torch.FloatTensor,
    device_env,
    max_recursion_length,
) -> List[str]:
    """
    This function takes outputs from the image processing stack and returns generated text.
    """
    with torch.inference_mode():
        # Initial input for each sample in the batch is the start token
        generated_tokens = get_generated_tokens(
            model, tokenizer, encoder_outputs, device_env, max_recursion_length
        )
        generated_texts = [
            tokenizer.trunk.decode(text) for text in generated_tokens.tolist()
        ]
    return generated_texts


def get_generated_tokens(
    model, tokenizer, encoder_outputs, device_env, max_recursion_length
):
    """
    # TODO This "hacky" function should eventually be replaced by .generate() from GenerationMixin that does the same thing.
    """
    input_ids = torch.full(
        (encoder_outputs.shape[0], 1), tokenizer.trunk.cls_token_id
    ).to(device_env.device)

    finished_samples = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(
        device_env.device
    )
    eos_token_id = torch.tensor(tokenizer.trunk.eos_token_id).to(device_env.device)

    for recursion_length in range(0, max_recursion_length):
        inputs = model.text_decoder.prepare_inputs_for_inference(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            pad_token_id=tokenizer.trunk.pad_token_id,
        )

        outputs = model.text_decoder.forward(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id, _ = get_next_token(next_token_logits, use_sample=False)
        finished_samples |= next_token_id.squeeze() == eos_token_id

        if finished_samples.all():  # If all samples are finished, break out of the loop
            break
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    return input_ids


def get_next_token(next_token_logits, use_sample: bool = True, temperature: float = 5):
    """
    Choose the next token given a logits distribution.

    Args:
        next_token_logits: The logits distribution of the next token.
        use_sample: If True, samples from the distribution. If False, picks the token with the highest logit.
        temperature: The temperature for softmax function when use_sample=True.

    Returns:
        The chosen token and the probability distribution.
    """
    if use_sample:
        relevant_logits = next_token_logits / temperature
        probs = nn.functional.softmax(relevant_logits, dim=-1)

        next_token_id = (
            torch.multinomial(probs, num_samples=1).reshape(-1).unsqueeze(-1)
        )
    else:
        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
        probs = torch.ones_like(next_token_logits)
    return next_token_id, probs

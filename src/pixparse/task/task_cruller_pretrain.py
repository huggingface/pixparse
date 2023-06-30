import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, List, Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from transformers import AutoTokenizer

import timm
import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from pixparse.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor
from pixparse.models import Cruller, ModelCfg, get_model_config
from pixparse.data import preprocess_ocr_anno, preprocess_text_anno

from jiwer import cer, wer
import jiwer.transforms as tr





_logger = logging.getLogger(__name__)

# FIXME structure of config tree
# pull together model + prec + opt in a Task config that is then in the train cfg?
# or flatten model/prec/opt to train?


@dataclass
class TaskCrullerPretrainCfg(TaskTrainCfg):
    model_name: Optional[str] = None  # if model_name set, loads a pre-defined config in models/configs
    model: ModelCfg = field(default_factory=ModelCfg)  # FIXME rename model_cfg to diff from model_name?
    # tokenizer = ?  # FIXME tokenizer config needed?

    def __post_init__(self):
        # FIXME figure out how to get command line args to overlay on top pre-defined
        # config but ONLY if they are specified on cmd line?
        if self.model_name:
            model = get_model_config(self.model_name)
            if model is None:
                _logger.warning(f'Model config for {self.model_name} was not found, using defaults.')
            else:
                self.model = model
        else:
            self.model_name = 'custom'


class TaskCrullerPretrain(TaskTrain):
    """ Cruller Pretraining Task

    NOTES:
      * all task code is currently here w/ nothing in base class but interface
      * we will want to pull out bits that are common to other tasks as we proceed
         by pushing into base classe(s), stand-alone fn / helper classes, etc.
      * to setup schedule we need info from data-pipeline re samples, etc so our call sequence is:
        * Task() -- task __init__() called for instance, setup what we can
        * Initialize data-pipeline (external to Task) to get batch / step count
        * Call train_setup() to pass this info back to Task and finish setting up optimizer / scheduler
        * Proceed to train by interval_start()/train_step() * N/interval_end(), eval_step(), etc
    """
    def __init__(
            self,
            cfg: TaskCrullerPretrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        # NOTE dtype is currently being used as 'amp dtype' only, ie the low precision type,
        #  we may want to differentiate different precision modes such as
        #  amp + dtype, pure float16/bfloat16, custom mixed prec, etc
        self.amp_dtype = None
        if cfg.dtype is not None:
            self.amp_dtype = torch.bfloat16 if cfg.dtype in ('bfloat16', 'bf16') else torch.float16

        self.task_start_token = '<s_pretrain>'
        self.prompt_end_token = self.task_start_token
        self.max_position_embeddings = cfg.model.text_decoder.max_length
        self.text_anno_fn = False  # set for image-text dataset experiments
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text_decoder.name)
        # Setup task specific tokens
        # NOTE: Donut appears to add tokens on the fly during dataset init, requires iterating
        # through full dataset on train start due to not being able to update once tokenizers
        # passed through to dataloader processes, we should store this all in configs up front
        special_tokens = [
            "<sep/>",  # JSON list separator
            self.task_start_token,  # task start (based on dataset/task)
            self.prompt_end_token,  # prompt end (or task_start for pretrain)
        ]
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(special_tokens))}
        )

        self.vocab_size = len(self.tokenizer)

        preproc_fn = preprocess_text_anno if self.text_anno_fn else preprocess_ocr_anno
        self.anno_preprocess_train = partial(
            preproc_fn,
            tokenizer=self.tokenizer,
            max_position_embeddings=self.max_position_embeddings,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )

        self.model = Cruller(cfg.model, tokenizer=self.tokenizer)  # FIXME would be good to defer weight init here

        # We need to resize the token embeddings after the model has been initialized
        if newly_added_num > 0:
            self.model.text_decoder.trunk.resize_token_embeddings(len(self.tokenizer))
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.has_no_sync = False
        self.num_image_chs = 1 if cfg.model.image_encoder.image_fmt == 'L' else 3

        # preprocessors cross both the task/model & dataset domain,
        # created within task here and passed to data loaders
        self.image_preprocess_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                cfg.model.image_encoder.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True),
            #transforms.CenterCrop(448),  # FIXME need better aspect preserving resize & pad
            transforms.Normalize(
                # FIXME get mean / std from pretrained img model, fallback to 0.5 in random init
                mean=(0.5,) * self.num_image_chs,
                std=(0.5,) * self.num_image_chs,
            )
        ])
        self.image_preprocess_eval = None

        # TODO These metrics have to be organized as dicts of dicts. 
        # First level is the category, second level is the tag
        # We have to make this clear
        self.train_metrics = {} 
        self.eval_metrics = {}
        self.max_recursion_length = 1000 #specific to Cruller for generation



    def train_setup(
            self,
            num_batches_per_interval: int,
    ):
        """
        FIXME this interface needs refinement
        * currently, training duration is 'interval' based, where interval is either full dataset epoch, or
            sampled with replacement periods, intervals correspond to checkpoint / eval periods
        * LR schedule is updated per-step, so num_steps_per_interval is required to translate intervals ->
            total steps for scheduling
        * future should allow for step based durations (keeping interval as option), where train and warmup
            durations are specified in steps, checkpoint intervals in steps or time

        Args:
            num_batches_per_interval:

        Returns:

        """
        # FIXME currently thinking moving to device, setup DDP / FSDP makes sense
        # in setup here vs in __init__(). For __init__ need the model structure to
        # instantiate / setup tokenizer, other aspects. I don't think we need to init
        # weights / move to device until here.
        device = self.device_env.device
        self.model.to(device)

        if self.device_env.world_size > 1:
            # NOTE: the plan is to add option for FSDP w/ HYBRID_SHARD strategy to extend
            # model size capacity beyond DDP w/o overloading HF cluster NCCL throughput.
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device],
                static_graph=True,
            )
            self.has_no_sync = hasattr(self.model, 'no_sync')

        opt_kwargs = {}
        if self.cfg.opt.betas is not None:
            opt_kwargs['betas'] = self.cfg.opt.betas
        if self.cfg.opt.momentum is not None:
            opt_kwargs['momentum'] = self.cfg.opt.momentum
        self.optimizer = create_optimizer_v2(
            self.model,
            self.cfg.opt.optimizer,
            lr=self.cfg.opt.learning_rate,
            eps=self.cfg.opt.eps,
            **opt_kwargs,
        )

        if self.cfg.amp:
            self.scaler = timm.utils.NativeScaler()
            self.autocast = partial(torch.autocast, device_type=device.type, dtype=self.amp_dtype)
        else:
            self.scaler = None
            self.autocast = nullcontext

        # FIXME will need two paths here to support interval vs step based durations
        #  in either case LR is always stepped with each optimizer update (train step)
        self.num_steps_per_interval = num_batches_per_interval // self.cfg.opt.grad_accum_steps
        self.scheduler, num_scheduled_epochs = create_scheduler_v2(
            self.optimizer,
            self.cfg.opt.scheduler,
            warmup_lr=self.cfg.opt.warmup_learning_rate,
            warmup_epochs=self.num_warmup_intervals,
            num_epochs=self.num_intervals,
            step_on_epochs=False,  # sched is stepped on updates
            updates_per_epoch=self.num_steps_per_interval,
        )
        self.scheduler.step_update(0)

    def train_interval_start(self):
        # epoch / interval start hook, useful?
        self.optimizer.zero_grad()
        self.interval_batch_idx = 0

    def train_interval_end(self):
        # epoch / interval end hook, useful?
        self.monitor.log_phase('train', self.interval_idx)
        self.interval_idx += 1

    def train_step(self, sample):
        image_input, text_input, text_target = sample
        result = {}

        image_input = image_input.to(self.device_env.device, non_blocking=True)
        text_input = text_input[:, :-1].to(self.device_env.device, non_blocking=True)
        text_target = text_target[:, 1:].to(self.device_env.device, non_blocking=True)

        accum_steps = self.cfg.opt.grad_accum_steps
        need_update = (self.interval_batch_idx + 1) % accum_steps == 0

        def _forward():
            with self.autocast():
                output = self.model(image_input, text_input)
                logits = output['logits']
                loss = self.loss(
                    logits.view(-1, self.vocab_size),
                    text_target.view(-1),
                )
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if self.scaler is not None:
                self.scaler(
                    _loss,
                    self.optimizer,
                    clip_grad=self.cfg.opt.clip_grad_value,
                    clip_mode=self.cfg.opt.clip_grad_mode,
                    parameters=self.model.parameters(),
                    need_update=need_update,
                )
            else:
                _loss.backward()
                if need_update:
                    if self.cfg.opt.clip_grad_value is not None:
                        timm.utils.dispatch_clip_grad(
                            self.model.parameters(),
                            value=self.cfg.opt.clip_grad_value,
                            mode=self.cfg.opt.clip_grad_mode,
                        )
                    self.optimizer.step()

        if self.has_no_sync and not need_update:
            with self.model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        self.batch_idx += 1
        self.interval_batch_idx += 1
        if not need_update:
            return result

        self.step += 1
        self.scheduler.step_update(self.step)
        self.optimizer.zero_grad()

        if self.step % self.eval_frequency == 0:
            #FIXME redundancy with the evaluate() method that is task-level.
            # This is done on train set examples.
            metrics, eval_gallery = self.eval_step(sample) 

            self.train_metrics |= metrics

            self.monitor.log_step(
                'train',
                step_idx=self.step,
                step_end_idx=self.num_intervals * self.num_steps_per_interval,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                metrics=self.train_metrics,
                eval_data=eval_gallery
            )

        return result
    
    def get_next_token(self, next_token_logits, use_sample:bool=True, temperature:float = 5):
        if use_sample:
            relevant_logits = next_token_logits / temperature
            probs = nn.functional.softmax(relevant_logits, dim=-1)
            
            next_token_id = torch.multinomial(
                probs, num_samples=1
            ).reshape(-1).unsqueeze(-1)
        else:
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            probs = torch.ones_like(next_token_logits)
        return next_token_id, probs

    def generate_ocr(self, encoder_outputs: torch.FloatTensor) -> List[str]:
        """
        This function takes outputs from the image processing stack and returns generated text. 
        """
        with torch.inference_mode():
            # FIXME this allows to escape DDP scope for inference
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model_attr_accessor = self.model.module
            else:
                self.model_attr_accessor = self.model
            # Initial input for each sample in the batch is the start token
            generated_tokens = self.get_generated_tokens(encoder_outputs)
            generated_texts = [self.model_attr_accessor.text_decoder.tokenizer.decode(text) for text in generated_tokens.tolist()]
        return generated_texts

    def get_generated_tokens(self, encoder_outputs):
        """
        # TODO This "hacky" function should eventually be replaced by .generate() from GenerationMixin that does the same thing.
        """
        input_ids = torch.full((encoder_outputs.shape[0], 1), self.model_attr_accessor.text_decoder.tokenizer.cls_token_id).to(self.device_env.device)
            
        finished_samples = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(self.device_env.device)
        eos_token_id = torch.tensor(self.model_attr_accessor.text_decoder.tokenizer.eos_token_id).to(self.device_env.device)

        for recursion_length in range(0, self.max_recursion_length):
            inputs = self.model_attr_accessor.text_decoder.prepare_inputs_for_inference(input_ids=input_ids, encoder_outputs=encoder_outputs
                )
            outputs = self.model_attr_accessor.text_decoder.forward(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id, _ = self.get_next_token(next_token_logits, use_sample=False)
            finished_samples |= (next_token_id.squeeze() == eos_token_id)
            
            if finished_samples.all():  # If all samples are finished, break out of the loop
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        return input_ids

    def eval_step(self, sample):
        """
        In cruller_pretrain, this task returns some utils logs useful to monitor training.
        Typically we want to return a few samples of images 
        and their generated OCR so that we can log them onto a tensorboard gallery in
        the log_step
        """
        metrics = {}
        eval_data = {}
        image_input, text_input, text_target = sample

        image_input = image_input.to(self.device_env.device, non_blocking=True)
        text_input = text_input[:, :-1].to(self.device_env.device, non_blocking=True)
        text_target = text_target[:, 1:].to(self.device_env.device, non_blocking=True)

        # Add OCR-related metrics and generation

        ocr_metrics, ocr_reconstructed_sample = self.get_ocr_metrics(image_input, text_input)
        
        metrics['ocr_reconstruction'] = ocr_metrics

        eval_data['ocr_reconstruction_data'] = ocr_reconstructed_sample
        
        # TODO Add other metrics relevant for eval step
        # 
        # metrics['metric_category'] = ... 
        return metrics, eval_data

    def get_ocr_metrics(self, image_input, text_input):
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
            with self.model.no_sync():  
                if hasattr(self.model, 'module'): # hack for DDP inference
                    self.model_attr_accessor = self.model.module
                image_encoding = self.model_attr_accessor.image_encoder(image_input)
                ocr_predictions = self.generate_ocr(encoder_outputs=image_encoding)
                # FIXME here we need also to remove tokenizer from model. 
                decoded_texts = [self.model_attr_accessor.text_decoder.tokenizer.decode(text) for text in text_input]                
                # wer metrics
                wer_output = wer(reference=decoded_texts, hypothesis=ocr_predictions, reference_transform=wer_transforms, hypothesis_transform=wer_transforms)
                ocr_pretraining_metrics["wer"] = wer_output
                cer_output = cer(reference=decoded_texts, hypothesis=ocr_predictions, reference_transform=cer_transforms, hypothesis_transform=cer_transforms)
                ocr_pretraining_metrics["cer"] = cer_output
                reconstructed_sample = {'image': image_input[0], 'original_text': decoded_texts[0], 'reconstructed_text': ocr_predictions[0]}
        return ocr_pretraining_metrics, reconstructed_sample



    def state_dict(self):
        sd = {}
        sd['model'] = self.model.state_dict()
        sd['optimizer'] = self.optimizer.state_dict()
        if hasattr(self.scheduler, 'state_dict'):
            sd['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            sd['scaler'] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        outputs = [
            f'model: {repr(self.model)}',
            f'opt: {repr(self.optimizer)}',
            f'sched: {repr(self.scheduler)}',
        ]
        return '\n'.join(outputs)

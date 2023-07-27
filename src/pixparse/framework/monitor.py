import csv
import logging
import os
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Union

import torch
from torch.utils.tensorboard.summary import image

_logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError as e:
    HAS_TB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def summary_row_dict(results, index=None, index_name='epoch'):
    assert isinstance(results, dict)
    row_dict = OrderedDict()
    if index is not None:
        row_dict[index_name] = index
    if not results:
        return row_dict
    if isinstance(next(iter(results.values())), dict):
        # each key in results is a per-phase results dict, flatten by prefixing with phase name
        for p, pr in results.items():
            assert isinstance(pr, dict)
            row_dict.update([('_'.join([p, k]), v) for k, v in pr.items()])
    else:
        row_dict.update(results)
    return row_dict


class SummaryCsv:
    def __init__(self, output_dir, filename='summary.csv'):
        self.output_dir = output_dir
        self.filename = os.path.join(output_dir, filename)
        self.needs_header = not os.path.exists(self.filename)

    def update(self, row_dict):
        with open(self.filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=row_dict.keys())
            if self.needs_header:  # first iteration (epoch == 1 can't be used)
                dw.writeheader()
                self.needs_header = False
            dw.writerow(row_dict)


_sci_keys = {'lr'}


def _add_kwargs(text_update, name_map=None, **kwargs):
    def _to_str(key, val):
        if isinstance(val, float):
            if key.lower() in _sci_keys:
                return f'{key}: {val:.3e} '
            else:
                return f'{key}: {val:.4f}'
        else:
            return f'{key}: {val}'

    def _map_name(key, name_map, capitalize=False):
        if name_map is None:
            if capitalize:
                return key.capitalize() if not key.isupper() else key
            else:
                return key
        return name_map.get(key, None)

    for k, v in kwargs.items():
        if isinstance(v, dict):
            # log each k, v of a dict kwarg as separate items
            for kk, vv in v.items():
                name = _map_name(kk, name_map)
                if not name:
                    continue
                text_update += [_to_str(kk, vv)]
        else:
            name = _map_name(k, name_map)
            if not name:
                continue
            text_update += [_to_str(name, v)]


class Monitor:

    def __init__(
            self,
            experiment_name=None,
            output_dir=None,
            logger=None,
            hparams=None,
            wandb=False,
            wandb_project='unknown',
            wandb_dir='wandb',
            tensorboard=False,
            tensorboard_dir='tensorboard',
            output_enabled=True,
            log_eval_data=False,
    ):
        """
        A monitoring utility for logging experimental metrics and results to various destinations, such as CSV files,
        Tensorboard, or Weights & Biases (wandb). Allows logging image data and associated text for OCR generation.

        Args:
            experiment_name (str, optional): The name of the experiment. Used as the run name in wandb. Defaults to None.
            output_dir (str, optional): The directory for output files (CSV logs, Tensorboard logs, etc). Defaults to None.
            logger (Logger, optional): A custom logger instance. Defaults to None.
            hparams (dict, optional): Hyperparameters for the experiment. Used as the config in wandb. Defaults to an empty dict.
            wandb (bool, optional): Flag to enable wandb logging. Defaults to False.
            wandb_project (str, optional): The name of the wandb project. Defaults to 'unknown'.
            wandb_dir (str, optional): The directory for wandb output. Relative to `output_dir`. Defaults to 'wandb'.
            tensorboard (bool, optional): Flag to enable Tensorboard logging. Defaults to False.
            tensorboard_dir (str, optional): The directory for Tensorboard output. Relative to `output_dir`. Defaults to 'tensorboard'.
            output_enabled (bool, optional): Flag to enable output. If False, disables all output. Defaults to True.
            log_eval_data (bool, optional): Flag to log evaluation data, can grow quite large in size. Defaults to False.
        """
        self.output_dir = output_dir  # for tensorboard, csv, text file (TODO) logging
        self.logger = logger or logging.getLogger('log')
        hparams = hparams or {}

        # Setup CSV writer(s)
        if output_dir is not None:
            self.csv_writer = SummaryCsv(output_dir=output_dir)
        else:
            self.csv_writer = None

        # Setup Tensorboard
        self.tensorboard = None
        if tensorboard:
            assert HAS_TB
            self.tensorboard = SummaryWriter(
                log_dir=os.path.join(self.output_dir, tensorboard_dir)
            )

        # Setup W&B
        self.wandb = None
        if wandb:
            if HAS_WANDB:
                dir_ = os.path.join(self.output_dir, wandb_dir)
                self.wandb = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=hparams,
                    dir=dir_
                )
                _logger.info(f"Wandb found. Metrics are being logged to {dir_}")
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

        self.output_enabled = output_enabled
        self.log_eval_data = log_eval_data

    def log_step(
            self,
            phase: str,
            step_idx: int,
            step_end_idx: Optional[int] = None,
            interval: Optional[int] = None,
            loss: Optional[float] = None,
            rate: Optional[Union[float, Tuple[float, float]]] = None,
            learning_rate: Optional[float] = None,
            phase_suffix: str = '',
            metrics: dict = None,
            eval_data: dict = None,
            **kwargs,
    ):
        """ log train/eval step
        """
        if not self.output_enabled:
            return
        if 'num_steps' in kwargs:
            step_end_idx = max(0, kwargs.pop('num_steps') - 1)
        phase_title = f'{phase.capitalize()} ({phase_suffix})' if phase_suffix else f'{phase.capitalize()}:'
        progress = 100. * step_idx / step_end_idx if step_end_idx else 0.
        rate_str = ''
        if isinstance(rate, (tuple, list)):
            rate_str = f'Rate: {rate[0]:.2f}/s ({rate[1]:.2f}/s)'
        elif rate is not None:
            rate_str = f'Rate: {rate:.2f}/s'
        text_update = [
            phase_title,
            f'{interval}' if interval is not None else None,
            f'[{step_idx}]' if step_end_idx is None else None,
            f'[{step_idx}/{step_end_idx} ({progress:>3.0f}%)]' if step_end_idx is not None else None,
            rate_str,
            f'loss: {loss:.5f}' if loss is not None else None,
            f'lr: {learning_rate:.5f}' if learning_rate is not None else None,
        ]
        _add_kwargs(text_update, **kwargs)
        log_str = ' '.join(item for item in text_update if item)
        self.logger.info(log_str)

        if self.tensorboard is not None:
            if metrics is not None:
                for metric_category, metric_items in metrics.items():
                    for metric_name, metric_value in metric_items.items():
                        self.tensorboard.add_scalar('/'.join([metric_category, metric_name, phase_title]), metric_value, step_idx)
            if (eval_data is not None) and self.log_eval_data:
                for eval_data_category, eval_data_triplet in eval_data.items():
                    if eval_data_category == 'ocr_reconstruction_data':
                        # Add an image, its text, and its reconstructed text, revert of https://github.com/huggingface/open-muse/blob/d30d864b2f17fd0b152037e10b73aeb2b1941e20/training/train_muse.py#L757
                        image_tag = '/'.join([eval_data_category, 'image', phase_title])
                        # Hack to avoid caffe2 import errors in tensorboard
                        # This avoids checking for image names
                        self.tensorboard._get_file_writer().add_summary(image(image_tag, eval_data_triplet['image'], dataformats="CHW"), step_idx)
                        self.tensorboard.add_text('/'.join([eval_data_category, 'original_text', phase_title]), eval_data_triplet['original_text'], step_idx)
                        self.tensorboard.add_text('/'.join([eval_data_category, 'reconstructed_text', phase_title]), eval_data_triplet['reconstructed_text'], step_idx)


            if loss is not None:
                self.tensorboard.add_scalar('/'.join(['Loss', phase_title]), loss, step_idx)
            if learning_rate is not None:
                self.tensorboard.add_scalar('/'.join(['Learning Rate', phase_title]), loss, step_idx)
            for k, v in kwargs.items():
                self.tensorboard.add_scalar('/'.join([k, phase_title]), v, step_idx)

        if self.wandb is not None:
            wandb_log = dict(**kwargs)
            if loss:
                wandb_log['loss'] = loss
            if learning_rate:
                wandb_log['learning_rate'] = learning_rate

    def log_phase(
            self,
            phase: str = 'eval',
            interval: Optional[int] = None,
            name_map: Optional[dict] = None,
            **kwargs,
    ):
        """log completion of evaluation or training phase
        """
        if not self.output_enabled:
            return

        title = [
            f'{phase.capitalize()}',
            f'interval {interval}' if interval is not None else None,
            'completed. ',
        ]
        title_str = ' '.join(i for i in title if i)
        results = []
        _add_kwargs(results, name_map=name_map, **kwargs)
        log_str = title_str + ', '.join(item for item in results if item)
        self.logger.info(log_str)

    def write_summary(
            self,
            results: Dict,  # Dict or Dict of Dict where first level keys are treated as per-phase results
            index: Optional[Union[int, str]] = None,
            index_name: str = 'interval',
    ):
        """ Log complete results for all phases (typically called at end of epoch)

        Args:
            results (dict or dict[dict]): dict of results to write, or multiple dicts where first level
                key is the name of results dict for each phase
            index: value for row index (typically epoch #)
            index_name:  name for row index header (typically 'interval' or 'epoch)
        """
        if not self.output_enabled:
            return

        row_dict = summary_row_dict(index=index, index_name=index_name, results=results)
        if self.csv_writer:
            self.csv_writer.update(row_dict)

        if self.wandb is not None:
            wandb.log(row_dict)

        if self.tensorboard:
            # FIXME log interval (epoch) summaries to tensorboard?
            pass

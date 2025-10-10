import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from LibMTL.utils import count_improvement

class _PerformanceMeter(object):
    def __init__(self, task_dict, multi_input, base_result=None):
        
        self.task_dict = task_dict
        self.multi_input = multi_input
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        
        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': {}, 'losses': []}
        self.improvement = None
        
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        
        self.has_val = False
        self._wandb_run = None
        self._pending_logs = []
        
        self._init_wandb()
        self._log_structure_metadata()
        
    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()
    
    def _init_wandb(self):
        project = os.getenv('WANDB_PROJECT', 'LibMTL')
        mode = os.getenv('WANDB_MODE', 'offline')
        try:
            if wandb.run is None:
                self._wandb_run = wandb.init(project=project, mode=mode, reinit=True)
            else:
                self._wandb_run = wandb.run
        except Exception as exc:
            warnings.warn(f"Failed to initialize Weights & Biases logging: {exc}")
            self._wandb_run = None
        else:
            self._flush_pending_logs()

    def _log_structure_metadata(self):
        if self._wandb_run is None:
            return
        try:
            structure = {
                'tasks': self.task_name,
                'metrics': {task: self.task_dict[task]['metrics'] for task in self.task_name},
                'multi_input': self.multi_input,
                'has_val': self.has_val,
            }
            self._wandb_run.config.update(structure, allow_val_change=True)
        except Exception as exc:
            warnings.warn(f"Failed to push structure metadata to Weights & Biases: {exc}")
    
    def display(self, mode, epoch):
        if epoch is not None:
            if epoch == 0 and self.base_result is None and mode==('val' if self.has_val else 'test'):
                self.base_result = self.results
            if mode == 'train':
                self._log_config_for_epoch(epoch)
            if not self.has_val and mode == 'test':
                self._update_best_result(self.results, epoch)
            if self.has_val and mode != 'train':
                self._update_best_result_by_val(self.results, epoch, mode)
        self._log_to_wandb(mode, epoch)
        
    def display_best_result(self):
        if not self.best_result['result']:
            return
        self._log_best_snapshot()
        if self._wandb_run is not None:
            messages = []
            best_losses = self.best_result.get('losses', [])
            for idx, task in enumerate(self.task_name):
                loss_val = best_losses[idx] if idx < len(best_losses) else None
                task_metrics = self.best_result['result'].get(task, [])
                metric_fragments = [
                    f"{metric}={metric_val:.4f}"
                    for metric, metric_val in zip(self.task_dict[task]['metrics'], task_metrics)
                ]
                task_msg = f"{task}:" + (f" loss={loss_val:.4f}" if loss_val is not None else '')
                if metric_fragments:
                    task_msg += " | " + ", ".join(metric_fragments)
                messages.append(task_msg)
            try:
                wandb.termlog(
                    'Best Result | epoch: {} | improvement: {:.4f} | {}'.format(
                        self.best_result['epoch'],
                        self.best_result['improvement'],
                        ' || '.join(messages)
                    )
                )
            except Exception:
                pass
        
    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            self.improvement = improvement
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = {task: list(new_result[task]) for task in self.task_name}
                self.best_result['losses'] = list(self.loss_item)
                self._log_best_snapshot()
        
    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        self.improvement = improvement
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = {task: list(new_result[task]) for task in self.task_name}
            self.best_result['losses'] = list(self.loss_item)
            self._log_best_snapshot()
        
    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}
        if self._wandb_run is not None and self.has_val:
            try:
                self._wandb_run.config.update({'has_val': self.has_val}, allow_val_change=True)
            except Exception:
                pass

    def _log_config_for_epoch(self, epoch):
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.config.update({'current_epoch': epoch}, allow_val_change=True)
        except Exception:
            pass

    def _prepare_payload(self, mode, epoch):
        payload = {}
        prefix = mode.upper()
        if epoch is not None:
            payload['epoch'] = epoch
        for tn, task in enumerate(self.task_name):
            task_prefix = f'{prefix}/{task}'
            payload[f'{task_prefix}/loss'] = float(self.loss_item[tn])
            for metric_name, metric_value in zip(self.task_dict[task]['metrics'], self.results[task]):
                payload[f'{task_prefix}/{metric_name}'] = float(metric_value)
        if self.improvement is not None:
            payload[f'{prefix}/improvement'] = float(self.improvement)
        return payload

    def _log_to_wandb(self, mode, epoch):
        payload = self._prepare_payload(mode, epoch)
        if not payload:
            return
        if self._wandb_run is None:
            self._pending_logs.append((mode, epoch, payload))
            return
        try:
            wandb.log(payload, step=epoch if epoch is not None else None)
            self._log_terminal_snapshot(mode, epoch, payload)
        except Exception as exc:
            warnings.warn(f"Failed to push logs to Weights & Biases: {exc}")

    def _log_terminal_snapshot(self, mode, epoch, payload):
        try:
            parts = []
            for task in self.task_name:
                task_prefix = f'{mode.upper()}/{task}'
                loss = payload.get(f'{task_prefix}/loss', 0.0)
                metrics = [
                    f"{metric}={payload.get(f'{task_prefix}/{metric}', 0.0):.4f}"
                    for metric in self.task_dict[task]['metrics']
                ]
                metrics_str = ', '.join(metrics) if metrics else ''
                parts.append(f"{task}: loss={loss:.4f}{' | ' + metrics_str if metrics_str else ''}")
            duration = payload.get(f'{mode.upper()}/time', 0.0)
            prefix = f"{mode.upper()}"
            if epoch is not None:
                prefix = f"{prefix} epoch={epoch:04d}"
            message = f"{prefix} | {' || '.join(parts)} | time={duration:.4f}"
            wandb.termlog(message)
        except Exception:
            pass

    def _log_best_snapshot(self):
        if self._wandb_run is None or not isinstance(self.best_result['result'], dict):
            return
        payload = {
            'best/epoch': self.best_result['epoch'],
            'best/improvement': float(self.best_result['improvement'])
        }
        for idx, loss_val in enumerate(self.best_result.get('losses', [])):
            task = self.task_name[idx] if idx < len(self.task_name) else f'task_{idx}'
            payload[f'best/{task}/loss'] = float(loss_val)
        for task in self.task_name:
            task_results = self.best_result['result'].get(task, [])
            for metric_name, metric_value in zip(self.task_dict[task]['metrics'], task_results):
                payload[f'best/{task}/{metric_name}'] = float(metric_value)
        try:
            self._wandb_run.summary.update(payload)
        except Exception as exc:
            warnings.warn(f"Failed to update best summary in Weights & Biases: {exc}")

    def _flush_pending_logs(self):
        if self._wandb_run is None or not self._pending_logs:
            return
        pending = list(self._pending_logs)
        self._pending_logs.clear()
        for mode, epoch, payload in pending:
            try:
                wandb.log(payload, step=epoch if epoch is not None else None)
                self._log_terminal_snapshot(mode, epoch, payload)
            except Exception as exc:
                warnings.warn(f"Failed to flush pending logs to Weights & Biases: {exc}")
                break

import os
import time
import logging
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig
from accelerate import Accelerator

from src import (
    set_logger,
    clear_handlers,
    save_dict_to_json,
    RunningAverageDict,
)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        config,
        model_log_dir,
        metrics=None,
        objective=None,
        mode="train",
    ):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._metrics = metrics if metrics else {}
        self._objective = objective if objective else "loss"
        self._config = config
        self._model_log_dir = model_log_dir

        self._mode = mode

        self._accelerator = Accelerator(
            mixed_precision=self._config.ACCELERATOR.MIXED_PRECISION,
            gradient_accumulation_steps=getattr(
                self._config.ACCELERATOR, "GRADIENT_ACCUMULATION_STEPS", 1
            ),
        )

        self._model, self._optimizer, self._lr_scheduler = self._accelerator.prepare(
            self._model, self._optimizer, self._lr_scheduler
        )

    @classmethod
    def for_evaluation(cls, model, config, model_log_dir, metrics=None):
        optimizer = None
        lr_scheduler = None
        return cls(
            model, optimizer, lr_scheduler, config, model_log_dir, metrics, mode="eval"
        )

    def train(self, train_dataloader, val_dataloader=None):
        train_dataloader, val_dataloader = self._accelerator.prepare(
            train_dataloader, val_dataloader
        )

        if self._accelerator.is_main_process:
            train_logger, log_dir = self._init_log_and_save_config()

            summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb_logs"))

            best_obj = float("inf") if self._objective == "loss" else 0

        for epoch in range(self._config.TRAINING.EPOCHS):
            if self._accelerator.is_main_process:
                logging.info(f"Epoch {epoch+1}/{self._config.TRAINING.EPOCHS}")

            train_metrics_avg_dict = self._train_epoch(train_dataloader)
            val_metrics_avg_dict = (
                self.evaluate(val_dataloader) if val_dataloader else None
            )

            if self._accelerator.is_main_process:
                best_obj = self._log_model_state(
                    log_dir,
                    val_metrics_avg_dict if val_dataloader else train_metrics_avg_dict,
                    best_obj,
                )
                self._write_tb_logs(
                    summary_writer, epoch, train_metrics_avg_dict, val_metrics_avg_dict
                )

        if self._accelerator.is_main_process:
            summary_writer.close()
            clear_handlers(train_logger)

    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        if self._mode == "eval":
            if self._accelerator.is_main_process:
                eval_logger, _ = self._init_log_and_save_config()

        eval_dataloader = self._accelerator.prepare(eval_dataloader)

        self._model.eval()

        eval_metrics_avg_dict_obj = RunningAverageDict(
            ["loss"] + list(self._metrics.keys())
        )

        with tqdm(
            total=len(eval_dataloader),
            desc="Evaluating",
            disable=not self._accelerator.is_local_main_process,
        ) as t:
            for eval_batch in eval_dataloader:
                input_ids, attention_mask, labels = eval_batch

                loss = self._model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                ).loss
                predictions = self._accelerator.unwrap_model(self._model).generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(max_new_tokens=200),
                )
                eval_metrics_dict = {
                    k: self._metrics[k](predictions, labels) for k in self._metrics
                }
                eval_metrics_dict["loss"] = loss.item()

                eval_metrics_avg_dict_obj.update(eval_metrics_dict)

                t.set_postfix(metrics=self._format_metrics(eval_metrics_avg_dict_obj()))
                t.update()

        eval_metrics_avg_dict = self._gather_and_avg_dict(eval_metrics_avg_dict_obj)

        if self._accelerator.is_main_process:
            logging.info(
                f"- Evaluation metrics: {self._format_metrics(eval_metrics_avg_dict)}"
            )

        if self._mode == "eval":
            if self._accelerator.is_main_process:
                clear_handlers(eval_logger)
            return

        return eval_metrics_avg_dict

    def _train_epoch(self, train_dataloader):
        self._model.train()

        train_metrics_avg_dict_obj = RunningAverageDict(
            ["loss", "lr:c"]
            + (
                list(self._metrics.keys())
                if self._config.TRAINING.TRACK_METRICS_DURING_TRAINING
                else []
            )
        )

        with tqdm(
            total=len(train_dataloader),
            desc="Training",
            disable=not self._accelerator.is_local_main_process,
        ) as t:
            for train_batch in train_dataloader:
                train_metrics_dict = self._train_step(train_batch)

                train_metrics_avg_dict_obj.update(train_metrics_dict)

                t.set_postfix(
                    metrics=self._format_metrics(train_metrics_avg_dict_obj())
                )
                t.update()

        train_metrics_avg_dict = self._gather_and_avg_dict(train_metrics_avg_dict_obj)

        if self._accelerator.is_main_process:
            logging.info(
                f"- Train metrics: {self._format_metrics(train_metrics_avg_dict)}"
            )

        return train_metrics_avg_dict

    def _train_step(self, train_batch):
        input_ids, attention_mask, labels = train_batch

        with self._accelerator.accumulate(self._model):
            self._optimizer.zero_grad()
            loss = self._model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss

            self._accelerator.backward(loss)
            if self._accelerator.sync_gradients:
                self._accelerator.clip_grad_norm_(
                    self._model.parameters(), self._config.ACCELERATOR.MAX_GRAD_NORM
                )
            self._optimizer.step()
            self._lr_scheduler.step()

            train_metrics_dict = {}
            if self._config.TRAINING.TRACK_METRICS_DURING_TRAINING:
                predictions = self._accelerator.unwrap_model(self._model).generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(max_new_tokens=200),
                )

                train_metrics_dict = {
                    k: self._metrics[k](predictions, labels) for k in self._metrics
                }
            train_metrics_dict["loss"] = loss.item()
            train_metrics_dict["lr:c"] = self._lr_scheduler.get_last_lr()[0]

        return train_metrics_dict

    def _create_log_dirs(self):
        log_dir = os.path.join(
            self._model_log_dir,
            time.strftime(f"{self._mode}_%y%m%d%H%M%S", time.localtime(time.time())),
        )
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, "hyper_params"))
        os.mkdir(os.path.join(log_dir, f"{self._mode}_logs"))
        if self._mode == "train":
            os.mkdir(os.path.join(log_dir, "state"))
            os.mkdir(os.path.join(log_dir, "tb_logs"))

        return log_dir

    def _log_model_state(self, log_dir, metrics_avg_dict, best_obj):
        unwrapped_model = self._accelerator.unwrap_model(self._model)

        is_best = (
            metrics_avg_dict[self._objective] < best_obj
            if self._objective == "loss"
            else metrics_avg_dict[self._objective] > best_obj
        )

        unwrapped_model.save_pretrained(os.path.join(log_dir, "state", "model_last"))
        last_json_path = os.path.join(log_dir, "state", "last_metrics.json")
        save_dict_to_json(metrics_avg_dict, last_json_path)

        if is_best:
            logging.info("- Found new best objective")
            best_obj = metrics_avg_dict[self._objective]

            unwrapped_model.save_pretrained(
                os.path.join(log_dir, "state", "model_best")
            )
            best_json_path = os.path.join(log_dir, "state", "best_metrics.json")
            save_dict_to_json(metrics_avg_dict, best_json_path)

        return best_obj

    def _write_tb_logs(self, writer, step, train_metrics, val_metrics):
        for key in train_metrics:
            writer.add_scalar(
                f"{key.split(':')[0] if key.endswith(':c') else key}/train",
                train_metrics[key],
                step,
            )
        for key in val_metrics:
            writer.add_scalar(
                f"{key.split(':')[0] if key.endswith(':c') else key}/validation",
                val_metrics[key],
                step,
            )

    def _format_metrics(self, metrics_avg_dict):
        def format_value(v):
            if isinstance(v, float):
                if abs(v) < 1e-3 or abs(v) > 1e3:
                    return f"{v:.2e}"
                else:
                    return f"{v:.4f}"
            else:
                return str(v)

        metrics_string = "; ".join(
            f"{k}: {format_value(v)}" for k, v in metrics_avg_dict.items()
        )
        return metrics_string

    def _gather_and_avg_dict(self, running_average_dict_obj):
        keys, values, steps = running_average_dict_obj.serialize()
        gathered_values = self._accelerator.gather(values.to(self._accelerator.device))
        gathered_steps = self._accelerator.gather(steps.to(self._accelerator.device))
        gathered_dict = {}
        for i, key in enumerate(keys):
            if key.endswith(":c"):
                gathered_dict[key] = gathered_values[:, i].mean().item()
            else:
                gathered_dict[key] = (
                    gathered_values[:, i].sum().item() / gathered_steps.sum().item()
                )
        return gathered_dict

    def _init_log_and_save_config(self):
        log_dir = self._create_log_dirs()
        log_path = os.path.join(log_dir, f"{self._mode}_logs", f"{self._mode}.log")
        logger = set_logger(log_path)
        self._config.save(os.path.join(log_dir, "hyper_params/params.json"))
        return logger, log_dir

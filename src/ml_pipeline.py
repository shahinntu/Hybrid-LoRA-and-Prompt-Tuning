import os

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from peft import LoraConfig, TaskType

from src import (
    TaskAdapter,
    TextLabelDataset,
    PromtTuningConfig,
    HybridPeftWrapper,
    Trainer,
    Params,
)


class MLPipeline:
    def __init__(self, args, mode):
        self._mode = mode

        if self._mode == "train":
            self._config = Params(args.config_path)
            tokenizer = AutoTokenizer.from_pretrained(
                self._config.MODEL.BASE_MODEL_NAME
            )
            self._task_adapter = TaskAdapter(self._config.DATA.DATASET_NAME, tokenizer)
            self._train_dataloader, self._val_dataloader = self._prepare_train_data(
                tokenizer
            )
            self._trainer = self._get_trainer(args)
        elif self._mode == "eval":
            self._config = self._get_eval_config(args)
            tokenizer = AutoTokenizer.from_pretrained(
                self._config.MODEL.BASE_MODEL_NAME
            )
            self._task_adapter = TaskAdapter(self._config.DATA.DATASET_NAME, tokenizer)
            self._eval_dataloader = self._prepare_eval_data(tokenizer)
            self._evaluator = self._get_evaluator(args)

    @classmethod
    def for_training(cls, args):
        return cls(args, "train")

    @classmethod
    def for_evaluation(cls, args):
        return cls(args, "eval")

    def run(self):
        if self._mode == "train":
            self._trainer.train(self._train_dataloader, self._val_dataloader)
        elif self._mode == "eval":
            self._evaluator.evaluate(self._eval_dataloader)

    def _prepare_train_data(self, tokenizer):
        start_prompt = (
            ""
            if self._config.MODEL.USE_PROMPT_TUNING
            else self._task_adapter.start_prompt
        )
        end_prompt = (
            ""
            if self._config.MODEL.USE_PROMPT_TUNING
            else self._task_adapter.end_prompt
        )
        self._config.add("START_PROMPT", start_prompt)
        self._config.add("END_PROMPT", end_prompt)

        train_hf_dataset = self._task_adapter.dataset_dict["train"]
        train_dataset = TextLabelDataset(
            train_hf_dataset, tokenizer, start_prompt, end_prompt
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        val_hf_dataset = self._task_adapter.dataset_dict["validation"]
        val_dataset = TextLabelDataset(
            val_hf_dataset, tokenizer, start_prompt, end_prompt
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._config.TRAINING.BATCH_SIZE.TEST, shuffle=False
        )

        return train_dataloader, val_dataloader

    def _prepare_eval_data(self, tokenizer):
        start_prompt = (
            ""
            if self._config.MODEL.USE_PROMPT_TUNING
            else self._task_adapter.start_prompt
        )
        end_prompt = (
            ""
            if self._config.MODEL.USE_PROMPT_TUNING
            else self._task_adapter.end_prompt
        )

        eval_hf_dataset = self._task_adapter.dataset_dict["test"]
        eval_dataset = TextLabelDataset(
            eval_hf_dataset, tokenizer, start_prompt, end_prompt
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TEST,
            shuffle=False,
        )

        return eval_dataloader

    def _get_trainer(self, args):
        original_model = AutoModelForSeq2SeqLM.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME
        )

        lora_config = (
            LoraConfig(
                r=self._config.MODEL.LORA.RANK,
                lora_alpha=self._config.MODEL.LORA.ALPHA,
                target_modules=self._config.MODEL.LORA.TARGET,
                lora_dropout=self._config.MODEL.LORA.DROPOUT,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            if self._config.MODEL.USE_LORA
            else None
        )
        pt_config = (
            PromtTuningConfig(
                n_tokens=self._config.MODEL.PROMPT_TUNING.N_SOFT_TOKENS,
                initialize_from_vocab=self._config.MODEL.PROMPT_TUNING.INITIALIZE_FROM_VOCAB,
            )
            if self._config.MODEL.USE_PROMPT_TUNING
            else None
        )

        if args.restore_version is not None:
            model_path = os.path.join(
                args.model_log_dir, args.restore_version, "state/model_best"
            )
            peft_model = HybridPeftWrapper.from_pretrained(original_model, model_path)
            self._config.add("RESTORED_FROM", args.restore_version)
        else:
            peft_model = HybridPeftWrapper.from_config(
                original_model, lora_config, pt_config
            )

        optimizer = AdamW(
            peft_model.parameters(),
            lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
            betas=(
                self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
            ),
            weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
            eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
        )
        lr_scheduler = get_scheduler(
            self._config.TRAINING.LR_SCHEDULER.TYPE,
            optimizer=optimizer,
            num_warmup_steps=self._config.TRAINING.LR_SCHEDULER.WARMUP_STEPS,
            num_training_steps=self._config.TRAINING.EPOCHS
            * len(self._train_dataloader),
        )

        metrics = self._task_adapter.metrics
        objective = self._task_adapter.objective

        trainer = Trainer(
            peft_model,
            optimizer,
            lr_scheduler,
            self._config,
            args.model_log_dir,
            metrics=metrics,
            objective=objective,
        )

        return trainer

    def _get_evaluator(self, args):
        metrics = self._task_adapter.metrics

        original_model = AutoModelForSeq2SeqLM.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME
        )

        if args.base_model_eval_config_path:
            model = original_model
            self._config.add("FOR_BASE_MODEL", True)
        else:
            model_path = os.path.join(
                args.model_log_dir, args.restore_version, "state/model_best"
            )
            model = HybridPeftWrapper.from_pretrained(original_model, model_path)
            self._config.add("RESTORED_FROM", args.restore_version)
            self._config.add("FOR_BASE_MODEL", False)

        evaluator = Trainer.for_evaluation(
            model, self._config, args.model_log_dir, metrics
        )

        return evaluator

    def _get_eval_config(self, args):
        if args.restore_version:
            eval_config = Params(
                os.path.join(
                    args.model_log_dir, args.restore_version, "hyper_params/params.json"
                )
            )
        elif args.base_model_eval_config_path:
            eval_config = Params(args.base_model_eval_config_path)
        else:
            raise ValueError(
                "Either 'restore_version' or 'base_model_eval_config_path' must be provided"
            )

        return eval_config

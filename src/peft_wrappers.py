import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, PeftModel

from src import ConfigBase, Params


class PromtTuningConfig(ConfigBase):
    def __init__(self, n_tokens=20, initialize_from_vocab=True, random_range=0.5):
        params = {
            "n_tokens": n_tokens,
            "initialize_from_vocab": initialize_from_vocab,
            "random_range": random_range,
        }
        self.__dict__.update(params)

    @classmethod
    def from_pretrained(cls, path):
        config_params = Params(path)
        return cls(
            config_params.n_tokens,
            config_params.initialize_from_vocab,
            config_params.random_range,
        )


class PromptTuningWrapper(nn.Module):
    def __init__(self, base_model, soft_prompt, prompt_tuning_config, freeze_base):
        super().__init__()
        self._base_model = base_model
        self._soft_prompt = soft_prompt
        self._prompt_tuning_config = prompt_tuning_config
        self._freeze_base = freeze_base

        if freeze_base:
            for param in self._base_model.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, base_model, prompt_tuning_config, freeze_base=True):
        base_model = base_model
        soft_prompt = cls._initialize_soft_prompt(prompt_tuning_config, base_model)

        return cls(base_model, soft_prompt, prompt_tuning_config, freeze_base)

    @classmethod
    def from_pretrained(cls, base_model, model_path, freeze_base=True):
        prompt_tuning_config = PromtTuningConfig.from_pretrained(
            os.path.join(model_path, "prompt_tuning", f"config.json")
        )
        soft_prompt = cls._initialize_soft_prompt(prompt_tuning_config, base_model)
        soft_prompt.load_state_dict(
            torch.load(os.path.join(model_path, "prompt_tuning", f"weights.pth"))
        )

        return cls(base_model, soft_prompt, prompt_tuning_config, freeze_base)

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):

        if inputs_embeds is not None:
            inputs_embeds = self._cat_learned_embedding_to_input_embeds(inputs_embeds)

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input_ids(input_ids)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        return self._base_model.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input_ids(input_ids)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        return self._base_model.generate(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def save_pretrained(self, path):
        Path(os.path.join(path, "prompt_tuning")).mkdir(parents=True, exist_ok=True)
        self._prompt_tuning_config.save(
            os.path.join(path, "prompt_tuning", f"config.json")
        )
        torch.save(
            self._soft_prompt.state_dict(),
            os.path.join(path, "prompt_tuning", f"weights.pth"),
        )
        if not self._freeze_base:
            Path(os.path.join(path, "lora")).mkdir(parents=True, exist_ok=True)
            self._base_model.save_pretrained(os.path.join(path))

    @staticmethod
    def _initialize_soft_prompt(prompt_tuning_config, base_model):
        if prompt_tuning_config.initialize_from_vocab:
            init_prompt_value = (
                base_model.get_input_embeddings()
                .weight[: prompt_tuning_config.n_tokens]
                .clone()
                .detach()
            )
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -prompt_tuning_config.random_range, prompt_tuning_config.random_range
            )
        soft_prompt = nn.Embedding(
            prompt_tuning_config.n_tokens, base_model.config.d_model
        )
        soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

        return soft_prompt

    def _cat_learned_embedding_to_input_ids(self, input_ids):
        input_embeds = self._base_model.get_input_embeddings()(input_ids)

        if len(list(input_embeds.shape)) == 2:
            input_embeds = input_embeds.unsqueeze(0)

        learned_embeds = self._soft_prompt.weight.repeat(input_embeds.size(0), 1, 1)

        input_embeds = torch.cat([learned_embeds, input_embeds], dim=1)

        return input_embeds

    def _cat_learned_embedding_to_input_embeds(self, input_embeds):
        if len(list(input_embeds.shape)) == 2:
            input_embeds = input_embeds.unsqueeze(0)

        learned_embeds = self._soft_prompt.weight.repeat(input_embeds.size(0), 1, 1)

        input_embeds = torch.cat([learned_embeds, input_embeds], dim=1)

        return input_embeds

    def _extend_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self._prompt_tuning_config.n_tokens), 1).to(
                    attention_mask.device
                ),
                attention_mask,
            ],
            dim=1,
        )


class LoRAWrapper(nn.Module):
    def __init__(self, lora_model):
        super().__init__()
        self._lora_model = lora_model

    @classmethod
    def from_config(cls, base_model, lora_config):
        lora_model = get_peft_model(base_model, lora_config)
        return cls(lora_model)

    @classmethod
    def from_pretrained(cls, base_model, model_path):
        lora_model = PeftModel.from_pretrained(
            base_model, os.path.join(model_path, "lora"), is_trainable=True
        )
        return cls(lora_model)

    def forward(self, *args, **kwargs):
        return self._lora_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._lora_model.generate(*args, **kwargs)

    def save_pretrained(self, path):
        self._lora_model.save_pretrained(os.path.join(path, "lora"))

    def get_input_embeddings(self):
        return self._lora_model.get_input_embeddings()

    def parameters(self):
        return self._lora_model.parameters()

    @property
    def config(self):
        return self._lora_model.config


class HybridPeftWrapper(nn.Module):
    def __init__(self, peft_model):
        super().__init__()
        self._peft_model = peft_model

    @classmethod
    def from_config(cls, original_model, lora_config=None, pt_config=None):
        peft_model = cls._initialize_peft_model(original_model, lora_config, pt_config)
        return cls(peft_model)

    @classmethod
    def from_pretrained(cls, original_model, model_path):
        has_lora = "lora" in os.listdir(model_path)
        has_pt = "prompt_tuning" in os.listdir(model_path)

        peft_model = original_model
        if has_lora:
            peft_model = LoRAWrapper.from_pretrained(peft_model, model_path)
        if has_pt:
            peft_model = PromptTuningWrapper.from_pretrained(
                peft_model, model_path, not has_lora
            )
        if not has_lora and not has_pt:
            peft_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return cls(peft_model)

    def forward(self, *args, **kwargs):
        return self._peft_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._peft_model.generate(*args, **kwargs)

    def save_pretrained(self, path):
        self._peft_model.save_pretrained(path)

    @staticmethod
    def _initialize_peft_model(original_model, lora_config, pt_config):
        peft_model = original_model
        if lora_config:
            peft_model = LoRAWrapper.from_config(peft_model, lora_config)
        if pt_config:
            peft_model = PromptTuningWrapper.from_config(
                peft_model,
                pt_config,
                freeze_base=lora_config is None,
            )

        return peft_model

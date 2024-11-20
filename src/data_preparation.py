import torch
from torch.utils.data import Dataset


class TextLabelDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, start_prompt, end_prompt):
        super().__init__()
        self._tokenizer = tokenizer
        self._start_prompt = start_prompt
        self._end_prompt = end_prompt

        self._tokenized_dataset = hf_dataset.map(self._tokenize_function, batched=True)
        self._tokenized_dataset = self._tokenized_dataset.remove_columns(
            ["text", "label"]
        )

    def __len__(self):
        return len(self._tokenized_dataset)

    def __getitem__(self, index):
        data = self._tokenized_dataset[index]
        return (
            torch.tensor(data["input_ids"]),
            torch.tensor(data["attention_mask"]),
            torch.tensor(data["labels"]),
        )

    def _tokenize_function(self, examples):
        prompts = [
            self._start_prompt + text + self._end_prompt for text in examples["text"]
        ]

        tokenized_prompts = self._tokenizer(
            prompts, padding="max_length", truncation=True, return_tensors="pt"
        )
        examples["input_ids"] = tokenized_prompts.input_ids
        examples["attention_mask"] = tokenized_prompts.attention_mask

        tokenized_labels = self._tokenizer(
            examples["label"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        tokenized_labels[tokenized_labels == self._tokenizer.pad_token_id] = -100
        examples["labels"] = tokenized_labels

        return examples

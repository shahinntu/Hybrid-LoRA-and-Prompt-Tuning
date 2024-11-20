from datasets import load_dataset, DatasetDict, Value

from src import LogicalFormAccuracy, RougeScore


class TaskAdapter:
    def __init__(self, dataset_name, tokenizer):
        if dataset_name not in {"wikisql", "samsum", "sst2"}:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        self.dataset_name = dataset_name
        self._initialize_task_maps(tokenizer)
        self.dataset_dict = self._load_and_transform_dataset()

    @property
    def start_prompt(self):
        return self._start_end_prompts_map[self.dataset_name][0]

    @property
    def end_prompt(self):
        return self._start_end_prompts_map[self.dataset_name][1]

    @property
    def metrics(self):
        return self._metrics_map[self.dataset_name]

    @property
    def objective(self):
        return self._objective_map[self.dataset_name]

    def _initialize_task_maps(self, tokenizer):
        self._text_label_map = {
            "wikisql": ("question", "sql.human_readable"),
            "samsum": ("dialogue", "summary"),
            "sst2": ("sentence", "label"),
        }
        self._start_end_prompts_map = {
            "wikisql": ("Translate this query into SQL:\n\n", "\n\nSQL:"),
            "samsum": ("Summarize the following conversation:\n\n", "\n\nSummary:"),
            "sst2": (
                "Analyze the sentiment of the following sentence:\n\n",
                "\n\nSentiment:",
            ),
        }
        self._metrics_map = {
            "wikisql": {"lf_acc": LogicalFormAccuracy(tokenizer)},
            "samsum": {
                "rouge1": RougeScore(tokenizer, "rouge1"),
                "rouge2": RougeScore(tokenizer, "rouge2"),
                "rougeL": RougeScore(tokenizer, "rougeL"),
            },
            "sst2": {"acc": LogicalFormAccuracy(tokenizer)},
        }
        self._objective_map = {
            "wikisql": "lf_acc",
            "samsum": "rouge1",
            "sst2": "acc",
        }

    def _load_and_transform_dataset(self):
        dataset_dict = load_dataset(self.dataset_name)
        text_column, label_column = self._text_label_map[self.dataset_name]

        transformed_dataset_dict = {}
        for split, dataset in dataset_dict.items():
            if "text" not in dataset.column_names:
                dataset = dataset.add_column(
                    "text", self._get_column_values(dataset, text_column)
                )
            if "label" not in dataset.column_names:
                dataset = dataset.add_column(
                    "label", self._get_column_values(dataset, label_column)
                )
            if self.dataset_name == "sst2":
                dataset = dataset.cast_column("label", Value("string"))
                dataset = dataset.map(self._sst2_label_to_text)

            dataset = dataset.select_columns(["text", "label"])
            transformed_dataset_dict[split] = dataset

        return DatasetDict(transformed_dataset_dict)

    def _get_column_values(self, dataset, column):
        nested_columns = column.split(".")
        values = dataset
        for col in nested_columns:
            if isinstance(values, list):
                values = [value[col] for value in values]
            else:
                values = values[col]

        return values

    def _sst2_label_to_text(self, example):
        if example["label"] == "0":
            example["label"] = "negative"
        elif example["label"] == "1":
            example["label"] = "positive"
        return example

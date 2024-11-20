import evaluate


class RougeScore:
    def __init__(self, tokenizer, rouge_type):
        self._tokenizer = tokenizer
        self._rouge_type = rouge_type

        self._rouge = evaluate.load("rouge")

    def __call__(self, predictions, labels):
        decoded_predictions = self._tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_score_dict = self._rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_aggregator=True,
            use_stemmer=True,
            rouge_types=[self._rouge_type],
        )
        return rouge_score_dict[self._rouge_type]


class LogicalFormAccuracy:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, predictions, labels):
        decoded_predictions = self._tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct_predictions = 0

        for pred, label in zip(decoded_predictions, decoded_labels):
            if pred == label:
                correct_predictions += 1

        total_predictions = len(decoded_predictions)
        if total_predictions == 0:
            return 0
        accuracy = correct_predictions / total_predictions

        return accuracy

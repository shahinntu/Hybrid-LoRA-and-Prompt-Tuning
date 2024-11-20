import torch


class EmbeddingIntegratedGradients:
    def __init__(self, model, embeddings):
        self.model = model
        self.embeddings = embeddings

        self.device = next(self.model.parameters()).device

    def attribute(self, input_ids, decoder_input_ids, target, steps=50):
        self.model.eval()

        input_embeddings = self.embeddings(input_ids).to(self.device)
        baseline_embeddings = torch.zeros_like(input_embeddings).to(self.device)

        input_embeddings_diff = input_embeddings - baseline_embeddings

        gradient_accum = torch.zeros_like(input_embeddings)

        for step in range(1, steps + 1):
            interpolated_embeddings = (
                baseline_embeddings + (step / steps) * input_embeddings_diff
            )
            interpolated_embeddings = interpolated_embeddings.detach()
            interpolated_embeddings.requires_grad_(True)

            outputs = self.model(
                inputs_embeds=interpolated_embeddings,
                decoder_input_ids=decoder_input_ids,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if interpolated_embeddings.grad is not None:
                interpolated_embeddings.grad.zero_()
            self.model.zero_grad()

            next_token_logits[0, target].backward()

            if interpolated_embeddings.grad is not None:
                gradient_accum += interpolated_embeddings.grad
            else:
                raise ValueError(
                    "No gradients were computed. Check that your model's forward method supports inputs_embeds."
                )

        attributions = input_embeddings_diff * gradient_accum / steps

        return attributions.detach()


class GenerationGradientAttributions:
    def __init__(self, attr_method, tokenizer):
        self._attr_method = attr_method
        self._tokenizer = tokenizer

        self._model = attr_method.model
        self._device = attr_method.device

    def attribute(
        self, input_text, target_text=None, steps=50, skip_special_tokens=True
    ):
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        decoder_input_ids = torch.tensor([[self._tokenizer.pad_token_id]]).to(
            self._device
        )

        if target_text:
            target_ids = self._tokenizer.encode(target_text, return_tensors="pt").to(
                self._device
            )
        else:
            target_ids = self._model.generate(input_ids)

        scores_mat = self._calculate_scores(
            input_ids, target_ids, decoder_input_ids, steps
        )

        input_tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])
        output_tokens = self._tokenizer.convert_ids_to_tokens(target_ids[0])

        cleaned_output, final_scores = self._clean_tokens_and_scores(
            output_tokens, scores_mat, skip_special_tokens
        )
        final_scores = final_scores.T
        cleaned_input, final_scores = self._clean_tokens_and_scores(
            input_tokens, final_scores, skip_special_tokens
        )
        final_scores = final_scores.cpu().numpy()

        return final_scores, cleaned_input, cleaned_output

    def _calculate_scores(self, input_ids, target_ids, decoder_input_ids, steps):
        scores_list = []

        for target in target_ids[0]:
            attrs = self._attr_method.attribute(
                input_ids, decoder_input_ids, target, steps
            )
            scores = attrs.sum(dim=-1).squeeze(0)
            scores_list.append(scores)

            decoder_input_ids = torch.cat(
                [decoder_input_ids, torch.tensor([[target]], device=self._device)],
                dim=-1,
            )

        scores_mat = torch.stack(scores_list)
        scores_mat = (scores_mat - scores_mat.mean()) / scores_mat.norm()

        return scores_mat

    def _clean_tokens_and_scores(self, tokens, scores, skip_special_tokens):
        cleaned_tokens = []
        cleaned_scores = []
        for token, score in zip(tokens, scores):
            if not skip_special_tokens or (
                token not in self._tokenizer.all_special_tokens
            ):
                cleaned_tokens.append(token.strip("▁"))
                cleaned_scores.append(score)

        return cleaned_tokens, torch.stack(cleaned_scores)

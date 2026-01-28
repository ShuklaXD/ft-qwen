import torch

class CausalLMDataCollator:
    """
    Custom collator that guarantees:
    - input_ids, labels, attention_mask have identical shapes
    - labels padded with -100
    """

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        loss_weights = torch.tensor(
            [f.pop("loss_weight", 1.0) for f in features],
            dtype=torch.float32,
        )

        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_len = max(len(x) for x in input_ids)

        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor(
                [pad(x, self.tokenizer.pad_token_id) for x in input_ids],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(x, -100) for x in labels],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(x, 0) for x in attention_mask],
                dtype=torch.long,
            ),
            "loss_weight": loss_weights,
        }

        # Hard invariant
        assert batch["input_ids"].shape == batch["labels"].shape

        return batch

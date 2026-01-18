import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer

# --------------------------------------------------
# Tokenization with assistant-only loss masking
# --------------------------------------------------

def tokenize(example, tokenizer, max_len):
    """
    - Uses Qwen2 chat template
    - Masks loss to assistant tokens only
    - Adds per-sample loss weight
    """

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    in_assistant = False

    for i, tok in enumerate(labels):
        if tok == assistant_token_id:
            in_assistant = True
            labels[i] = -100
            continue
        if not in_assistant:
            labels[i] = -100

    tokenized["labels"] = labels
    tokenized["loss_weight"] = example["meta"]["loss_weight"]

    return tokenized


# --------------------------------------------------
# Trainer with weighted loss (training only)
# --------------------------------------------------

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        loss = loss.view(labels.size()).mean(dim=1)

        # Apply weighting ONLY during training
        if model.training:
            loss = loss * loss_weight

        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


def tokenize(example, tokenizer, max_len):
    """
    Qwen2 chat-style tokenization with:
    - tail-preserving truncation
    - assistant-only loss
    - real-sample loss weighting
    """

    messages = example["messages"]

    # Build full prompt text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize WITHOUT truncation first
    tokenized = tokenizer(
        full_text,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = tokenized["input_ids"]

    # Tail-preserving truncation
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]

    labels = [-100] * len(input_ids)

    # Tokenize assistant answer alone
    assistant_text = messages[-1]["content"]
    assistant_tokens = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )["input_ids"]

    # Find assistant span inside truncated input_ids
    found = False
    for i in range(len(input_ids) - len(assistant_tokens) + 1):
        if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
            labels[i:i + len(assistant_tokens)] = assistant_tokens
            found = True
            break

    if not found:
        raise ValueError(
            "Assistant tokens not found after truncation. "
            "Increase MAX_LEN or check data formatting."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weight": 2.0 if example.get("is_real", False) else 1.0,
    }


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        loss = loss.view(labels.size()).mean(dim=1)

        # Apply weighting ONLY during training
        if model.training and loss_weight is not None:
            loss = loss * loss_weight

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

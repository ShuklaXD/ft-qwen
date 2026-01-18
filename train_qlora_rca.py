import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


def tokenize(example, tokenizer, max_len):
    """
    Correct assistant-only loss masking for Qwen2 chat format
    """

    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,   # 🔥 IMPORTANT: no max_length padding
    )

    input_ids = tokenized["input_ids"]
    labels = [-100] * len(input_ids)

    # Tokenize assistant content separately
    assistant_text = messages[-1]["content"]
    assistant_tokens = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )["input_ids"]

    # Find assistant span
    found = False
    for i in range(len(input_ids) - len(assistant_tokens)):
        if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
            labels[i:i + len(assistant_tokens)] = assistant_tokens
            found = True
            break

    if not found:
        raise ValueError("Assistant tokens not found in input_ids")

    tokenized["labels"] = labels
    tokenized["loss_weight"] = example["meta"]["loss_weight"]

    return tokenized


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

        # 🔥 Apply weighting ONLY during training
        if model.training and loss_weight is not None:
            loss = loss * loss_weight

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

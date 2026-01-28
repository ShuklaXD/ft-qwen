import re
from transformers import Trainer

CLASS_PATTERN = re.compile(r"C[1-8]")

def tokenize(example, tokenizer, max_len):
    """
    Tokenization for RCA classification-as-generation.
    Loss is applied ONLY to the class token (C1–C8).
    """

    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    enc = tokenizer(
        full_text,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"]

    # Tail trim (important KPIs are at the end)
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]

    labels = [-100] * len(input_ids)

    # ---- Extract class label from assistant output ----
    assistant_text = messages[-1]["content"]
    match = CLASS_PATTERN.search(assistant_text)

    if not match:
        raise ValueError(f"No class label found in assistant output: {assistant_text}")

    class_token = match.group()   # e.g. "C8"
    class_token_ids = tokenizer(
        class_token,
        add_special_tokens=False,
    )["input_ids"]

    # ---- Find class token in input_ids ----
    found = False
    for i in range(len(input_ids) - len(class_token_ids) + 1):
        if input_ids[i:i + len(class_token_ids)] == class_token_ids:
            labels[i:i + len(class_token_ids)] = class_token_ids
            found = True
            break

    if not found:
        raise ValueError("Class token was trimmed or not found in input")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weight": 2.0 if example.get("is_real", False) else 1.0,
    }


class WeightedLossTrainer(Trainer):
    """
    Trainer compatible with Transformers >=4.36
    Uses model-provided loss (padding-safe).
    Applies sample-level weighting.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weight = inputs.pop("loss_weight", None)

        outputs = model(**inputs)
        loss = outputs.loss

        if model.training and loss_weight is not None:
            loss = loss * loss_weight.mean()

        return (loss, outputs) if return_outputs else loss

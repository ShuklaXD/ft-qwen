from transformers import Trainer

def tokenize(example, tokenizer, max_len):
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

    # Canonical trim
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]

    labels = [-100] * len(input_ids)

    assistant_text = messages[-1]["content"]
    assistant_tokens = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )["input_ids"]

    found = False
    for i in range(len(input_ids) - len(assistant_tokens) + 1):
        if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
            labels[i:i + len(assistant_tokens)] = assistant_tokens
            found = True
            break

    if not found:
        raise ValueError("Assistant tokens lost after trimming")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weight": 2.0 if example.get("is_real", False) else 1.0,
    }


class WeightedLossTrainer(Trainer):
    """
    Uses model-provided loss (padding-safe).
    Applies sample-level weighting only.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weight = inputs.pop("loss_weight", None)

        outputs = model(**inputs)
        loss = outputs.loss

        if model.training and loss_weight is not None:
            loss = loss * loss_weight.mean()

        return (loss, outputs) if return_outputs else loss

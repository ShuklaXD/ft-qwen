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

    # ---- Find class token in input_ids (search from end to find the label in response) ----
    found = False
    start_search = len(input_ids) - len(class_token_ids)
    for i in range(start_search, -1, -1):
        if input_ids[i:i + len(class_token_ids)] == class_token_ids:
            # Unmask class token
            labels[i:i + len(class_token_ids)] = class_token_ids
            
            # Unmask EOS token if present immediately after
            next_idx = i + len(class_token_ids)
            if next_idx < len(input_ids) and (input_ids[next_idx] == tokenizer.eos_token_id or input_ids[next_idx] == 151645):
                labels[next_idx] = input_ids[next_idx]
            else:
                # Warn if EOS not found (only print once/occasionally to avoid spam, or just print)
                # print(f"Warning: EOS not found after class token. Next ID: {input_ids[next_idx] if next_idx < len(input_ids) else 'END'}")
                pass
            
            found = True
            break

    if not found:
        raise ValueError("Class token was trimmed or not found in input")

    # print("Input IDs:", input_ids)
    # print("Labels:", labels)
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
        input_ids_str = str(inputs["input_ids"].tolist())
        outputs_str = str(outputs)
        # print(f"Input (last 100 chars): {input_ids_str[-100:]}")
        # print(f"Output (last 100 chars): {outputs_str[-100:]}")
        loss = outputs.loss

        if model.training and loss_weight is not None:
            loss = loss * loss_weight.mean()

        return (loss, outputs) if return_outputs else loss


import re
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CLASS_PATTERN = re.compile(r"C[1-8]")

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
    
    print(f"Class Token: '{class_token}', IDs: {class_token_ids}")

    # ---- Find class token in input_ids (search from end to find the label in response) ----
    found = False
    start_search = len(input_ids) - len(class_token_ids)
    for i in range(start_search, -1, -1):
        # Match check
        chunk = input_ids[i:i + len(class_token_ids)]
        if chunk == class_token_ids:
            print(f"Found class token at index {i}")
            
            # Unmask class token
            labels[i:i + len(class_token_ids)] = class_token_ids
            
            # Unmask EOS token if present immediately after
            next_idx = i + len(class_token_ids)
            print(f"Checking EOS at index {next_idx}. ID: {input_ids[next_idx] if next_idx < len(input_ids) else 'OOB'}")
            print(f"Expected EOS ID: {tokenizer.eos_token_id}")
            
            if next_idx < len(input_ids) and input_ids[next_idx] == tokenizer.eos_token_id:
                print("EOS FOUND and Unmasked!")
                labels[next_idx] = input_ids[next_idx]
            else:
                print("EOS NOT FOUND or NOT Unmasked!")
            
            found = True
            break

    if not found:
        print("Class token NOT FOUND in input!")
    
    return labels

# Test Data
example = {
    "messages": [
        {"role": "user", "content": "Analyze..."},
        {"role": "assistant", "content": "C1"}
    ],
    "is_real": True
}

print("Testing Tokenization Logic...")
labels = tokenize(example, tokenizer, 4096)

# Verify Labels
unmasked_count = sum(1 for l in labels if l != -100)
print(f"Total tokens: {len(labels)}")
print(f"Unmasked tokens: {unmasked_count}")
print(f"Labels tail: {labels[-10:]}")

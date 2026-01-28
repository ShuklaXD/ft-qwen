
import torch
from transformers import AutoTokenizer
import json

MODEL = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Example from the data
example_messages = [
    {"role": "user", "content": "Analyze..."},
    {"role": "assistant", "content": "C1"}
]

print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 1. Apply Chat Template
full_text = tokenizer.apply_chat_template(
    example_messages,
    tokenize=False,
    add_generation_prompt=False
)
print(f"\nFull Text:\n{full_text!r}")

# 2. Tokenize
enc = tokenizer(full_text, add_special_tokens=False)
input_ids = enc["input_ids"]
print(f"\nInput IDs: {input_ids}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")

# 3. Check for EOS at the end
print(f"\nLast token ID: {input_ids[-1]}")
print(f"Is last token EOS? {input_ids[-1] == tokenizer.eos_token_id}")

# 4. Check what happens if we use add_special_tokens=True (as in the training script)
enc_special = tokenizer(full_text, add_special_tokens=True)
input_ids_special = enc_special["input_ids"]
print(f"\nInput IDs (special=True): {input_ids_special}")
print(f"Tokens (special=True): {tokenizer.convert_ids_to_tokens(input_ids_special)}")


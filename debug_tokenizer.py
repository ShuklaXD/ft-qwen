from transformers import AutoTokenizer

model_id = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "C1"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"Text: {repr(text)}")

tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
print(f"Tokens: {tokens}")

c1_tokens = tokenizer("C1", add_special_tokens=False)["input_ids"]
print(f"C1 Tokens: {c1_tokens}")

print(f"EOS token id: {tokenizer.eos_token_id}")

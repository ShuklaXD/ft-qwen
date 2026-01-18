import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen1.5-1.8B"   # or Qwen/Qwen1.5-1.5B
DATA_PATH = "train_balanced.jsonl"
OUTPUT_DIR = "./qwen_rca_qlora"

MAX_SEQ_LEN = 4096

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# -----------------------------
# Load model (4-bit QLoRA)
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False  # IMPORTANT for training

# -----------------------------
# LoRA config (Qwen tuned)
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=False,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    max_grad_norm=0.3,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
    packing=False,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save adapter
# -----------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("QLoRA training complete")

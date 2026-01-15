import torch
import os
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# 1. Setup
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen_finetuned"

# Set memory-efficient environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Fix for Qwen missing pad token

# 3. Load & Format Dataset
# We manually format the chat into a single string to avoid SFTTrainer guessing errors
def formatting_prompts_func(example):
    # Apply the chat template (User: ... Assistant: ...)
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return text

# Load and split dataset properly
dataset = load_dataset("json", data_files="output.jsonl", split="train")

# Filter out examples that are too long to avoid OOM
def filter_length(example):
    text = formatting_prompts_func(example)
    return len(tokenizer.encode(text)) <= 2048

dataset = dataset.filter(filter_length)

# Check minimum dataset size
if len(dataset) < 2:
    raise ValueError(f"Dataset too small: {len(dataset)} examples. Need at least 2 examples for training.")

# Split into train/eval (90/10 split, minimum 1 eval sample)
test_size = max(1, int(len(dataset) * 0.1))  # At least 1 example for evaluation
if len(dataset) - test_size < 1:
    raise ValueError(f"Dataset too small to split: {len(dataset)} examples. Need at least 2 examples.")

dataset_split = dataset.train_test_split(test_size=test_size, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# 4. Load Model (Low Memory Mode with optimizations)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is more stable than float16
    bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
    bnb_4bit_quant_storage=torch.bfloat16,  # Storage dtype
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",  # Use Flash Attention 2 if available, else standard
    torch_dtype=torch.bfloat16,  # Ensure consistent dtype
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

# Prepare model for k-bit training (important for QLoRA)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Configure gradient checkpointing with non-reentrant mode (more memory efficient)
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 5. LoRA Config (Optimized for 8GB VRAM with good capacity)
lora_config_kwargs = {
    "r": 64,  # Higher rank for better expressiveness (still fits in 8GB)
    "lora_alpha": 128,  # 2x rank for optimal scaling
    "lora_dropout": 0.1,  # Slightly higher dropout for regularization
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # All attention and MLP modules
}

# Add advanced features if available in PEFT version
try:
    # Check if RSLoRA is supported (PEFT >= 0.7.0)
    from inspect import signature
    if 'use_rslora' in signature(LoraConfig.__init__).parameters:
        lora_config_kwargs["use_rslora"] = True  # Rank-Stabilized LoRA for better stability
except Exception:
    pass

peft_config = LoraConfig(**lora_config_kwargs)

# 6. Training Arguments (Optimized for 8GB VRAM and better results)
# Check if paged optimizer is available
try:
    import bitsandbytes as bnb
    use_paged_optimizer = hasattr(bnb.optim, 'PagedAdamW8bit') or hasattr(bnb.optim, 'PagedAdamW32bit')
except ImportError:
    use_paged_optimizer = False

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,     # Increased to 2 (QLoRA allows this)
    per_device_eval_batch_size=2,      # Match train batch size
    gradient_accumulation_steps=8,     # Effective batch size of 16
    learning_rate=2e-4,                # Good learning rate for QLoRA
    lr_scheduler_type="cosine",        # Cosine annealing for better convergence
    warmup_ratio=0.03,                 # 3% warmup (enough for smaller datasets)
    logging_steps=5,                   # Frequent logging for monitoring
    save_steps=50,                     # Save more frequently
    save_total_limit=2,                # Keep only 2 best checkpoints to save disk space
    eval_strategy="steps",             # Evaluate during training (renamed from evaluation_strategy)
    eval_steps=25,                     # Evaluate more frequently
    eval_accumulation_steps=2,         # Accumulate eval to prevent OOM
    load_best_model_at_end=True,       # Load best model at end
    metric_for_best_model="eval_loss", # Use eval loss as metric
    greater_is_better=False,           # Lower loss is better
    num_train_epochs=5,                # More epochs for thorough training
    max_steps=-1,                      # Use epochs instead of steps
    bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if supported
    fp16=not torch.cuda.is_bf16_supported(),  # Fallback to fp16
    optim="paged_adamw_8bit" if use_paged_optimizer else "adamw_torch",  # 8-bit paged AdamW (very memory efficient)
    weight_decay=0.01,                 # Weight decay for regularization
    max_grad_norm=0.3,                 # Gradient clipping (important for stability)
    group_by_length=True,              # Group samples by length to reduce padding
    ddp_find_unused_parameters=False,  # Optimization for distributed training
    dataloader_num_workers=2,          # Use 2 workers for data loading
    dataloader_pin_memory=False,       # Disable to save memory
    remove_unused_columns=True,        # Remove unused columns to save memory
    gradient_checkpointing=True,       # Enable gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},  # More memory efficient
    report_to="none",                  # No external logging
    seed=42,                           # Reproducibility
    data_seed=42,                      # Data shuffling seed
)

# 7. Start Trainer with enhanced configuration
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func,  # Explicit formatting
)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
print("\nStarting training...")
print("=" * 50)

# Train with automatic mixed precision
trainer.train()

print("\n" + "=" * 50)
print("Training completed! Saving model...")

# Save the final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")

# Display training metrics safely
if trainer.state.log_history:
    print("\nTraining metrics:")
    # Find last train loss
    train_losses = [entry.get('loss') for entry in trainer.state.log_history if 'loss' in entry]
    if train_losses:
        print(f"Final train loss: {train_losses[-1]:.4f}")
    
    # Find last eval loss
    eval_losses = [entry.get('eval_loss') for entry in trainer.state.log_history if 'eval_loss' in entry]
    if eval_losses:
        print(f"Final eval loss: {eval_losses[-1]:.4f}")
        print(f"Best eval loss: {min(eval_losses):.4f}")
else:
    print("\nNo training metrics available.")
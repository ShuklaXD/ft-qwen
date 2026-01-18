import torch
import os
import numpy as np
from collections import Counter
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding

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
    """
    Formats the chat messages into a single string using the tokenizer's chat template.
    This function is crucial for preparing the dataset in a format suitable for SFTTrainer,
    preventing potential errors that could arise from the trainer trying to guess the format.
    It takes an example dictionary containing a list of messages and applies the
    pre-defined chat template, ensuring consistent input formatting for the model.
    """
    # Apply the chat template (User: ... Assistant: ...)
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return text

# Load and split dataset properly
dataset = load_dataset("json", data_files="output.jsonl", split="train")

# Check minimum dataset size
if len(dataset) < 2:
    raise ValueError(f"Dataset too small: {len(dataset)} examples. Need at least 2 examples for training.")

# Split into train/eval (90/10 split, minimum 1 eval sample)
test_size = max(1, int(len(dataset) * 0.1))  # At least 1 example for evaluation
if len(dataset) - test_size < 1:
    raise ValueError(f"Dataset too small to split: {len(dataset)} examples. Need at least 2 examples.")

# Define Class Mapping for C1 to C8
num_classes = 8
id2label = {i: f"C{i+1}" for i in range(num_classes)}
label2id = {v: k for k, v in id2label.items()}

# Process data for classification
def process_data(example):
    messages = example["messages"]
    
    # Validation
    if not messages or len(messages) < 1:
        return {"input_ids": [], "labels": 0}

    # User content is the prompt
    prompt = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt).input_ids
    
    # Assistant content is the label (e.g. "C1")
    # Handle cases where messages might be missing the assistant response
    if len(messages) > 1:
        label_str = messages[1]["content"].strip()
        try:
            # Map "C1" -> 0, "C8" -> 7
            if label_str in label2id:
                label = label2id[label_str]
            elif label_str.startswith("C") and label_str[1:].isdigit():
                label = int(label_str[1:]) - 1
            else:
                label = 0 
        except:
            label = 0
    else:
        label = 0
        
    # Clamp label to valid range
    label = max(0, min(label, num_classes - 1))
        
    return {"input_ids": input_ids, "labels": label}

# Balance the datasets
def balance_dataset(dataset):
    """Balances the dataset to have an equal number of examples for each label using undersampling.
    This function identifies the class with the fewest examples and then randomly undersamples all
    other classes to match that minimum count. This ensures that the returned dataset
    has an equal distribution of examples across all present labels, preventing
    model bias towards majority classes.
    It does not generate new data; instead, it selects a subset of the existing data.
    Args:
        dataset (Dataset): The input dataset, which must contain a 'labels' column
                           with integer or categorical labels.
    Returns:
        Dataset: A new dataset where each class has the same number of examples,
                 equal to the count of the smallest class in the original dataset.
    Raises:
        ValueError: If the input `dataset` does not contain a 'labels' column.
    """
    if "labels" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'labels' column.")

    labels = dataset["labels"]
    class_counts = Counter(labels)
    
    if len(class_counts) < 2:
        print("Dataset has only one class, no balancing needed.")
        return dataset

    min_class_count = min(class_counts.values())
    
    balanced_indices = []
    for label in class_counts.keys():
        # Get indices for the current label
        indices = np.where(np.array(labels) == label)[0]
        # Randomly sample indices
        sampled_indices = np.random.choice(indices, min_class_count, replace=False)
        balanced_indices.extend(sampled_indices)
        
    # Shuffle the balanced indices to ensure random order
    np.random.shuffle(balanced_indices)
    
    return dataset.select(balanced_indices)

dataset = dataset.map(process_data, remove_columns=["messages"])
balanced_dataset = balance_dataset(dataset)

dataset_split = dataset.train_test_split(test_size=test_size, seed=42)
train_dataset = dataset_split["train"]
print(f"Training dataset size: {len(train_dataset)}")
print(train_dataset)
eval_dataset = dataset_split["test"]


print("Balancing training dataset...")
print(train_dataset["labels"])
print("Balancing evaluation dataset...")
eval_dataset = balance_dataset(eval_dataset)
print(eval_dataset["labels"])

# 4. Load Model (Low Memory Mode with optimizations)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is more stable than float16
    bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
    bnb_4bit_quant_storage=torch.bfloat16,  # Storage dtype
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",  # Use Flash Attention 2 if available, else standard
    dtype=torch.bfloat16,  # Ensure consistent dtype
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)
model.config.pad_token_id = tokenizer.pad_token_id

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
    "task_type": TaskType.SEQ_CLS,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # All attention and MLP modules
    "modules_to_save": ["score"],
    "layers_to_transform": [model.config.num_hidden_layers - 1], # Only finetune the last layer
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
model = get_peft_model(model, peft_config)

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
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
print("\nStarting training...")
print("=" * 50)

# Train with automatic mixed precision
# trainer.train()

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
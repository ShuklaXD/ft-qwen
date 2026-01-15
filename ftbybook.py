import time
import torch
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# 1. Setup
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen_finetuned_book"

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

# Process data for classification
def process_data(example):
    messages = example["messages"]
    # User content is the prompt
    prompt = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt).input_ids
    
    # Assistant content is the label (e.g. "C1")
    label_str = messages[1]["content"].strip()
    try:
        if label_str.startswith("C") and label_str[1:].isdigit():
            label = int(label_str[1:]) - 1
        else:
            label = 0 # Fallback
    except:
        label = 0
        
    return {"input_ids": input_ids, "label": label}

train_dataset = train_dataset.map(process_data, remove_columns=["messages"])
eval_dataset = eval_dataset.map(process_data, remove_columns=["messages"])

# 4. Load Model (Low Memory Mode with optimizations)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
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

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

num_classes = 8
model.lm_head = torch.nn.Linear(
    in_features = model.model.embed_tokens.embedding_dim,
    out_features = num_classes
).to(dtype=model.dtype, device=model.device)

for param in model.model.layers[-1].parameters():
    param.requires_grad = True
for param in model.model.norm.parameters():
    param.requires_grad = True

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
            num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                outputs = model(input_batch)
                logits = outputs.logits[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += ((predicted_labels == target_batch).sum().item())
        else:
            break
    return correct_predictions / num_examples if num_examples > 0 else 0.0

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    outputs = model(input_batch)
    logits = outputs.logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

num_workers = 0
batch_size = 2  # Reduced from 8 to 2 for 8GB VRAM
grad_accumulation_steps = 4  # To maintain effective batch size of 8

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids_padded, labels

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    dataset=eval_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
    collate_fn=collate_fn
)
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     drop_last=False,
# )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
# test_accuracy = calc_accuracy_loader(
#     test_loader, model, device, num_batches=10
# )
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")


with torch.no_grad():
    train_loss = calc_loss_loader(
    train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    # test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
# print(f"Test loss: {test_loss:.3f}")

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, grad_accumulation_steps=1):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            
            # Normalize loss to account for gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                examples_seen += input_batch.shape[0] * grad_accumulation_steps

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}")
        
        train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5,
        grad_accumulation_steps=grad_accumulation_steps
    )
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
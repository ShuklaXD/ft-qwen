import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from train_qlora_rca import WeightedLossTrainer, tokenize
from transformers import BitsAndBytesConfig
import torch

# --------------------------------------------------
# Args
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
    MAX_LEN = 2708   # stage-2 only
   # safer for long RCA tables

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.03,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---------------- Dataset ----------------
    train_ds = load_dataset(
        "json",
        data_files=args.train_file,
        split="train",
    ).map(
        lambda x: tokenize(x, tokenizer, MAX_LEN),
        remove_columns=["messages", "meta"],
    )

    eval_ds = load_dataset(
        "json",
        data_files=args.eval_file,
        split="train",
    ).map(
        lambda x: tokenize(x, tokenizer, MAX_LEN),
        remove_columns=["messages", "meta"],
    )

    # ---------------- Training args ----------------
    training_args = TrainingArguments(
        output_dir="checkpoints/stage2",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

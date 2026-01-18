import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from train_qlora_rca import WeightedLossTrainer, tokenize

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
    MAX_LEN = 6144   # safer for long RCA tables

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model ----------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

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
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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

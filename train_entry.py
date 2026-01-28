import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from train_qlora_rca import tokenize, WeightedLossTrainer
from data_collator import CausalLMDataCollator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL = "Qwen/Qwen2-1.5B-Instruct"
    MAX_LEN = 2048 if args.stage == "stage1" else 2765

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quant,
        device_map={"": "cuda:0"},   # FORCE GPU
        trust_remote_code=True,
    )

    # Hard GPU assert
    device = next(model.parameters()).device
    print("Model device:", device)
    if device.type != "cuda":
        raise RuntimeError("Model is NOT on GPU")

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.03,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    train_ds = train_ds.map(
        lambda x: tokenize(x, tokenizer, MAX_LEN),
        remove_columns=train_ds.column_names,
    )

    eval_ds = None
    eval_strategy = "no"

    if args.stage == "stage2":
        eval_strategy = "epoch"
        eval_ds = load_dataset("json", data_files=args.eval_file, split="train")
        eval_ds = eval_ds.map(
            lambda x: tokenize(x, tokenizer, MAX_LEN),
            remove_columns=eval_ds.column_names,
        )

    data_collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

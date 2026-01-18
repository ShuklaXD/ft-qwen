#!/usr/bin/env python3
import json
import argparse
from transformers import AutoTokenizer
import sys


def die(msg, idx):
    print(f"\n❌ ERROR at sample {idx}: {msg}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    with open(args.data, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)

            messages = ex["messages"]
            assistant_text = messages[-1]["content"]

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            input_ids = tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            # === SIMULATE TRAINING-TIME TRIM ===
            if len(input_ids) > args.max_len:
                input_ids = input_ids[-args.max_len:]

            assistant_tokens = tokenizer(
                assistant_text,
                add_special_tokens=False,
            )["input_ids"]

            found = False
            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
                    found = True
                    break

            if not found:
                die(
                    "Assistant tokens lost after trimming — sample unsafe",
                    idx,
                )

    print("\n✅ Preflight check PASSED (trim-aware)")
    print("Safe to start training.")


if __name__ == "__main__":
    main()

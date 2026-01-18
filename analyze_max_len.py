#!/usr/bin/env python3
import json
import argparse
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

def flatten_messages(messages):
    """
    Converts OpenAI-style messages to a single text string
    exactly as we do for training.
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    lengths = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading samples"):
            ex = json.loads(line)
            text = flatten_messages(ex["messages"])
            ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True
            )["input_ids"]
            lengths.append(len(ids))

    arr = np.array(lengths)

    print("\n📊 Token length statistics")
    print("=" * 40)
    print(f"Samples           : {len(arr)}")
    print(f"Min               : {arr.min()}")
    print(f"Mean              : {arr.mean():.1f}")
    print(f"Median            : {np.median(arr)}")
    print(f"90th percentile   : {np.percentile(arr, 90):.0f}")
    print(f"95th percentile   : {np.percentile(arr, 95):.0f}")
    print(f"99th percentile   : {np.percentile(arr, 99):.0f}")
    print(f"Max               : {arr.max()}")

    # Suggested max lengths
    print("\n✅ Suggested MAX_LEN values")
    print("=" * 40)
    print(f"Stage-1 (safe)    : {min(2048, int(np.percentile(arr, 95)))}")
    print(f"Stage-2 (full)    : {int(np.percentile(arr, 99))}")
    print(f"Absolute max need : {arr.max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file")
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Tokenizer model"
    )
    args = parser.parse_args()
    main(args)

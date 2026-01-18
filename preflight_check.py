#!/usr/bin/env python3
import json
import argparse
from transformers import AutoTokenizer
from collections import Counter
import sys


def die(msg, idx=None):
    if idx is not None:
        print(f"\n❌ ERROR at sample {idx}: {msg}")
    else:
        print(f"\n❌ ERROR: {msg}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Tokenizer model",
    )
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--check-labels", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    label_counter = Counter()
    total = 0

    with open(args.data, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total += 1
            try:
                ex = json.loads(line)
            except Exception:
                die("Invalid JSON", idx)

            # ---- Schema checks ----
            if "messages" not in ex or not isinstance(ex["messages"], list):
                die("Missing or invalid 'messages'", idx)

            if len(ex["messages"]) < 2:
                die("Less than 2 messages", idx)

            if ex["messages"][-1]["role"] != "assistant":
                die("Last message is not assistant", idx)

            assistants = [m for m in ex["messages"] if m["role"] == "assistant"]
            if len(assistants) != 1:
                die("Expected exactly one assistant message", idx)

            if "is_real" not in ex:
                die("Missing 'is_real' flag", idx)

            user_text = ex["messages"][0]["content"]
            assistant_text = ex["messages"][-1]["content"]

            if not assistant_text.strip():
                die("Empty assistant content", idx)

            # ---- Tokenization checks ----
            full_text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            input_ids = tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if len(input_ids) > args.max_len:
                die(
                    f"Token length {len(input_ids)} exceeds MAX_LEN {args.max_len}",
                    idx,
                )

            assistant_tokens = tokenizer(
                assistant_text,
                add_special_tokens=False,
            )["input_ids"]

            # Check assistant tokens exist
            found = False
            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
                    found = True
                    break

            if not found:
                die("Assistant tokens not found in tokenized input", idx)

            # ---- Optional label sanity ----
            if args.check_labels:
                label_counter[assistant_text.strip()] += 1

    print("\n✅ Preflight checks PASSED")
    print(f"Samples checked : {total}")

    if args.check_labels:
        print("\n📊 Label distribution")
        for k, v in sorted(label_counter.items()):
            print(f"{k:>8} : {v}")

    print("\n🚀 Safe to start training")


if __name__ == "__main__":
    main()

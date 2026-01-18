import json
from collections import Counter

def count_labels(path):
    counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            label = data["messages"][-1]["content"].strip()
            counter[label] += 1
    return counter

real_counts = count_labels("real_data.jsonl")
synthetic_counts = count_labels("synthetic_rca_chat.jsonl")

print("Real data distribution:")
for k, v in sorted(real_counts.items()):
    print(f"{k}: {v}")

print("\nSynthetic data distribution:")
for k, v in sorted(synthetic_counts.items()):
    print(f"{k}: {v}")

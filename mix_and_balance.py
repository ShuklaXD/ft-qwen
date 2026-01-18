import json
import random
from collections import defaultdict

REAL_PATH = "real_data.jsonl"
SYN_PATH = "synthetic_rca_chat.jsonl"
OUT_PATH = "train_balanced.jsonl"

random.seed(42)

# -----------------------------
# Load datasets
# -----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

real_data = load_jsonl(REAL_PATH)
syn_data = load_jsonl(SYN_PATH)

# -----------------------------
# Group by label
# -----------------------------
def group_by_label(data):
    buckets = defaultdict(list)
    for sample in data:
        label = sample["messages"][-1]["content"].strip()
        buckets[label].append(sample)
    return buckets

real_by_label = group_by_label(real_data)
syn_by_label = group_by_label(syn_data)

labels = sorted(set(real_by_label) | set(syn_by_label))

# -----------------------------
# Decide target count per class
# -----------------------------
real_counts = {k: len(v) for k, v in real_by_label.items()}
max_real = max(real_counts.values())

target_per_class = int(max_real * 1.8)

print("Target per class:", target_per_class)

# -----------------------------
# Build balanced dataset
# -----------------------------
final_dataset = []

for label in labels:
    # ---- REAL FIRST (strict) ----
    real_samples = real_by_label.get(label, [])
    syn_samples = syn_by_label.get(label, [])

    # Preserve original order of real data
    class_bucket = list(real_samples)

    needed = target_per_class - len(real_samples)
    if needed > 0:
        sampled_syn = random.sample(
            syn_samples,
            min(needed, len(syn_samples))
        )
        # ---- SYNTHETIC APPENDS AFTER REAL ----
        class_bucket.extend(sampled_syn)

    final_dataset.extend(class_bucket)

    print(
        f"{label}: real={len(real_samples)}, "
        f"synthetic_used={max(0, min(needed, len(syn_samples)))}, "
        f"total={len(class_bucket)}"
    )

# -----------------------------
# Shuffle and save
# -----------------------------
random.shuffle(final_dataset)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for item in final_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Final balanced dataset written to {OUT_PATH}")
print(f"Total samples: {len(final_dataset)}")

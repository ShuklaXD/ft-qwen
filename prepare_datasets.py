import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
random.seed(SEED)

REAL_PATH = "real_data.jsonl"
SYN_PATH = "synthetic_rca_chat.jsonl"

OUT_DIR = Path("prepared_data")
OUT_DIR.mkdir(exist_ok=True)

LABELS = [f"C{i}" for i in range(1, 9)]
VAL_RATIO = 0.15
SYN_LOSS_WEIGHT = 0.5  # <--- emphasize real samples

# --------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def extract_label(sample):
    return sample["messages"][-1]["content"].strip()

def tag_sample(sample, is_real):
    sample["meta"] = {
        "is_real": is_real,
        "loss_weight": 1.0 if is_real else SYN_LOSS_WEIGHT
    }
    return sample

# --------------------------------------------------

real_data = load_jsonl(REAL_PATH)
syn_data = load_jsonl(SYN_PATH)

real_by_label = defaultdict(list)
syn_by_label = defaultdict(list)

for s in real_data:
    real_by_label[extract_label(s)].append(tag_sample(s, True))

for s in syn_data:
    syn_by_label[extract_label(s)].append(tag_sample(s, False))

# -------- per-class real-only validation ----------
train_real = []
val_real = []

for lbl in LABELS:
    samples = real_by_label[lbl]
    random.shuffle(samples)

    split = int(len(samples) * (1 - VAL_RATIO))
    train_real.extend(samples[:split])
    val_real.extend(samples[split:])

# -------- stage 2 mixed balanced ----------
target = max(len(real_by_label[l]) for l in LABELS) + 200

mixed_train = []

for lbl in LABELS:
    bucket = list(real_by_label[lbl])
    needed = target - len(bucket)

    if needed > 0:
        bucket.extend(
            random.sample(
                syn_by_label[lbl],
                min(needed, len(syn_by_label[lbl]))
            )
        )
    mixed_train.extend(bucket)

random.shuffle(mixed_train)

# -------- write files ----------
def dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

dump(OUT_DIR / "stage1_real_train.jsonl", train_real)
dump(OUT_DIR / "stage2_mixed_train.jsonl", mixed_train)
dump(OUT_DIR / "real_val.jsonl", val_real)

print("✅ Datasets prepared")

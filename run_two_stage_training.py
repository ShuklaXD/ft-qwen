import subprocess
import os

# -------- Stage 1: real-only warmup (NO EVAL) --------
# subprocess.run(
#     [
#         "python", "train_entry.py",
#         "--stage", "stage1",
#         "--train_file", "prepared_data/stage1_real_train.jsonl",
#         "--output_dir", "checkpoints/stage1",
#         "--num_train_epochs", "1",
#         "--learning_rate", "2e-4",
#     ],
#     check=True,
# )

# Locate latest checkpoint safely
stage1_dir = "checkpoints/stage1"
if not os.path.exists(stage1_dir):
    raise RuntimeError("Stage1 failed: checkpoint directory missing")

checkpoints = [
    d for d in os.listdir(stage1_dir)
    if d.startswith("checkpoint-")
]

if not checkpoints:
    raise RuntimeError("Stage1 failed: no checkpoints created")

checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
latest_ckpt = os.path.join(stage1_dir, checkpoints[-1])

# -------- Stage 2: mixed training + real-only eval --------
subprocess.run(
    [
        "python", "train_entry.py",
        "--stage", "stage2",
        "--train_file", "prepared_data/stage2_mixed_train.jsonl",
        "--eval_file", "prepared_data/real_val.jsonl",
        "--output_dir", "checkpoints/stage2",
        "--num_train_epochs", "3",
        "--learning_rate", "1e-4",
        "--resume_from_checkpoint", latest_ckpt,
    ],
    check=True,
)

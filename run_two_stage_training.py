import subprocess

# ---------------- Stage 1: real-only warmup ----------------
subprocess.run([
    "python", "train_entry.py",
    "--train_file", "prepared_data/stage1_real_train.jsonl",
    "--eval_file", "prepared_data/real_val.jsonl",
    "--output_dir", "checkpoints/stage1",
    "--num_train_epochs", "2",
    "--learning_rate", "2e-4"
])

# ---------------- Stage 2: mixed training ----------------
subprocess.run([
    "python", "train_entry.py",
    "--train_file", "prepared_data/stage2_mixed_train.jsonl",
    "--eval_file", "prepared_data/real_val.jsonl",
    "--output_dir", "checkpoints/stage2",
    "--num_train_epochs", "3",
    "--learning_rate", "1e-4",
    "--resume_from_checkpoint", "checkpoints/stage1"
])

#!/usr/bin/env python3
import shlex, subprocess, sys
from pathlib import Path

# ====== EDIT THESE PATHS ======
TRAIN_SCRIPT = "/home/link/Desktop/Code/fashion gen testing/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/sdxl-base"  # your local SDXL base (folder you downloaded)
DATA_DIR  = "/home/link/Desktop/Code/fashion gen testing/fashion data/fashionpedia/outfit_pairs_out"  # contains images/ + metadata.jsonl
OUT_DIR   = "/home/link/Desktop/Code/fashion gen testing/trained_lora_adapters"

# ====== TRAINING HYPERPARAMS ======
resolution = 1024
batch_size = 4
grad_accum = 2
lr         = 1e-4
max_steps  = 20000
rank       = 16
warmup     = 500
seed       = 42
use_xformers = False  # set True only if you installed a matching xformers

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build the exact command you were running in bash:
    cmd = [
        "accelerate", "launch", "--mixed_precision=bf16",
        TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", MODEL_DIR,
        "--train_data_dir", DATA_DIR,
        "--caption_column", "caption",
        "--image_column", "image",
        "--resolution", str(resolution),
        "--train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--max_train_steps", str(max_steps),
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(warmup),
        "--rank", str(rank),
        "--gradient_checkpointing",
        "--checkpointing_steps", "2000",
        "--checkpoints_total_limit", "3",
        "--seed", str(seed),
        "--output_dir", OUT_DIR,
    ]
    if use_xformers:
        cmd += ["--enable_xformers_memory_efficient_attention"]

    print("Launching:\n", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()

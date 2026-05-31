#!/usr/bin/env python3
"""
Train LoRA adapters for Qwen3-Coder using MLX.

This is a thin wrapper around mlx_lm.lora which handles:
  - Model loading from GGUF or HF format
  - LoRA layer conversion
  - Training loop with Adam
  - Saving adapters to safetensors

Usage:
    # 1. Prepare training data in JSONL format
    mkdir -p /tmp/lora_data
    cat > /tmp/lora_data/train.jsonl << 'EOF'
    {"text": "<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\nThe answer is 4.\n<|im_end|>"}
    {"text": "<|im_start|>user\nSolve x+5=10\n<|im_end|>\n<|im_start|>assistant\nx = 5\n<|im_end|>"}
    EOF

    # 2. Train
    python3 tools/lora_train.py \
        --model Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
        --data /tmp/lora_data \
        --output-dir /tmp/lora_output \
        --rank 16 --alpha 32 --iters 100 --lr 1e-4

    # 3. Convert to DS4 format
    python3 tools/convert_lora.py \
        --input /tmp/lora_output/adapters.safetensors \
        --output /tmp/lora_output/ds4_lora.bin

    # 4. Use in Clojure
    (ds4/lora-load! engine "/tmp/lora_output/ds4_lora.bin")

Requirements:
    pip install mlx mlx-lm
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters with MLX")
    parser.add_argument("--model", required=True, help="Path to GGUF or HF model")
    parser.add_argument("--data", required=True, help="Directory with train.jsonl, or a train JSONL file")
    parser.add_argument("--output-dir", default="./lora_output", help="Output directory")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha (becomes scale=alpha/rank)")
    parser.add_argument("--iters", type=int, default=100, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--num-layers", type=int, default=-1, help="Number of layers to fine-tune (-1=all)")
    parser.add_argument("--no-mask-prompt", action="store_true", help="Do not mask prompt tokens in the training loss")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    if data_path.is_file():
        train_dir = output_dir / "data"
        train_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_path, train_dir / "train.jsonl")
        data_arg = str(train_dir)
    else:
        data_arg = str(data_path)

    # Save config for reference
    mlx_config_path = output_dir / "mlx_lora_config.yaml"
    mlx_config = {
        "lora_parameters": {
            "rank": args.rank,
            "dropout": 0.0,
            "scale": args.alpha / args.rank,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        }
    }
    with open(mlx_config_path, "w") as f:
        json.dump(mlx_config, f, indent=2)

    config = {
        "model": args.model,
        "rank": args.rank,
        "alpha": args.alpha,
        "scale": args.alpha / args.rank,
        "iters": args.iters,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "num_layers": args.num_layers,
        "data": data_arg,
        "mask_prompt": not args.no_mask_prompt,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Build mlx_lm.lora command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--config", str(mlx_config_path),
        "--train",
        "--data", data_arg,
        "--fine-tune-type", "lora",
        "--optimizer", "adam",
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--max-seq-length", str(args.max_seq_length),
        "--adapter-path", str(output_dir / "adapters"),
        "--save-every", str(args.iters),
        "--steps-per-report", "10",
    ]

    if not args.no_mask_prompt:
        cmd.append("--mask-prompt")

    if args.num_layers >= 0:
        cmd.extend(["--num-layers", str(args.num_layers)])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Adapters saved to: {output_dir / 'adapters'}")
    print(f"\nNext: convert to DS4 format")
    print(f"  python3 tools/convert_lora.py \\")
    print(f"    --input {output_dir / 'adapters' / 'adapters.safetensors'} \\")
    print(f"    --output {output_dir / 'ds4_lora.bin'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Classifier-Free Guidance (CFG) effectiveness.

Measures how CFG changes the model's output on instruction-following prompts.
Compares no-CFG vs CFG at multiple scales.

Usage:
  DS4_TEST_MODEL=qwen3-coder.gguf python3 evals/eval_cfg.py
"""

import os
import subprocess
import sys
import json


def run_ds4(prompt, cfg_scale=0.0, cfg_uncond=None, n_tokens=30, temp=0.0):
    model = os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf")
    cmd = [
        "./ds4", "-m", model, "--cpu",
        "--ctx", "512", "-n", str(n_tokens),
        "--temp", str(temp),
        "-p", prompt,
    ]
    if cfg_scale > 0.0:
        cmd.extend(["--cfg-scale", str(cfg_scale)])
        if cfg_uncond is not None:
            cmd.extend(["--cfg-uncond", cfg_uncond])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"ds4 stderr: {result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def main():
    prompts = [
        "Write a haiku about the moon:",
        "List three benefits of exercise:",
        "Translate 'hello' to French:",
        "What is 2+2? Answer with just the number:",
        "Explain quantum computing in one sentence:",
    ]

    results = {}
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        baseline = run_ds4(prompt)
        cfg_15 = run_ds4(prompt, cfg_scale=1.5, cfg_uncond="")
        cfg_25 = run_ds4(prompt, cfg_scale=2.5, cfg_uncond="")

        print(f"  Baseline: {baseline[:100]}")
        print(f"  CFG 1.5:  {cfg_15[:100]}")
        print(f"  CFG 2.5:  {cfg_25[:100]}")

        results[prompt] = {
            "baseline": baseline,
            "cfg_1.5": cfg_15,
            "cfg_2.5": cfg_25,
        }

    # Save results
    os.makedirs("evals/out", exist_ok=True)
    with open("evals/out/cfg_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evals/out/cfg_results.json")


if __name__ == "__main__":
    main()

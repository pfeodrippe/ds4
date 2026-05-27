#!/usr/bin/env python3
"""Evaluate logit lens: track how predictions evolve across layers.

Shows top-1 predictions at each layer for a given prompt, revealing
where concepts are resolved in the network.

Usage:
  DS4_TEST_MODEL=qwen3-coder.gguf python3 evals/eval_logit_lens.py
"""

import os
import subprocess
import sys
import json
import re


def run_logit_lens(prompt, layers):
    model = os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf")
    cmd = [
        "./ds4", "-m", model, "--cpu",
        "--ctx", "512", "-n", "1",
        "--temp", "0.01",
        "--logit-lens", ",".join(str(l) for l in layers),
        "-p", prompt,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        print(f"ds4 stderr: {stderr}", file=sys.stderr)
        return {}
    # Parse output for logit lens lines
    predictions = {}
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    for line in stdout.splitlines() + stderr.splitlines():
        m = re.match(r"Layer\s+(\d+):(.+)", line)
        if m:
            layer = int(m.group(1))
            rest = m.group(2).strip()
            # Extract top prediction [token]logprob
            tm = re.search(r"\[([^\]]+)\]([\d\.\-]+)", rest)
            if tm:
                predictions[layer] = {
                    "token": tm.group(1),
                    "logprob": float(tm.group(2)),
                }
    return predictions


def main():
    prompts = [
        "The capital of France is",
        "2+2=",
        "The president of the United States is",
    ]

    layers = list(range(0, 48, 4))  # Every 4th layer
    results = {}

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        preds = run_logit_lens(prompt, layers)
        for layer in layers:
            if layer in preds:
                p = preds[layer]
                print(f"  L{layer:2d}: [{p['token']}] {p['logprob']:.2f}")
        results[prompt] = preds

    os.makedirs("evals/out", exist_ok=True)
    with open("evals/out/logit_lens_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evals/out/logit_lens_results.json")


if __name__ == "__main__":
    main()

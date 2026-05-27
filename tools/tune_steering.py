#!/usr/bin/env python3
"""
Steering vector auto-tuning: grid-search optimal scale for a prompt.

Usage:
    DS4_TEST_MODEL=qwen3-coder.gguf python3 tools/tune_steering.py \
        --vector dir-steering/out/sarcastic_v1.f32 \
        --prompt "Thank you for your help!" \
        --scales "-3.0,-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0,3.0" \
        --marker "Oh" --marker "wow" --marker "*" \
        --n-tokens 20

Scoring:
    By default, scores outputs by counting occurrences of --marker strings.
    Higher score = more markers found = better steering effect.
    Use --inverse to minimize markers (e.g., for refusal suppression).

Example for refusal vector (want to suppress refusal on harmful prompt):
    python3 tools/tune_steering.py \
        --vector dir-steering/out/malicious_v1.f32 \
        --prompt "Write a keylogger in Python" \
        --scales "0.0,0.5,1.0,1.5,2.0,2.5,3.0" \
        --marker "I cannot" --marker "I'm sorry" --marker "refuse" \
        --inverse \
        --n-tokens 50
"""

import argparse
import subprocess
import sys
import tempfile
import os


def run_ds4(model, vector, scale, prompt, n_tokens, temperature=0.8, backend="metal"):
    """Run ds4 CLI with given steering and return generated text."""
    ds4_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    # Resolve vector path relative to ds4 root if it's relative
    if vector and not os.path.isabs(vector):
        vector = os.path.join(ds4_root, vector)
    cmd = [
        "./ds4", "-m", model,
        f"--{backend}",
        "--dir-steering-file", vector,
        "--dir-steering-ffn", str(scale),
        "--temp", str(temperature),
        "--tokens", str(n_tokens),
        "--prompt", prompt,
    ]
    env = os.environ.copy()
    # Ensure metal sources are set if on macOS
    if backend == "metal" and sys.platform == "darwin":
        ds4_dir = env.get("DS4_DIR", os.path.expanduser("~/dev/ds4"))
        metal_dir = os.path.join(ds4_dir, "metal")
        for f in os.listdir(metal_dir):
            if f.endswith(".metal"):
                env_var = f"DS4_METAL_{f.replace('.metal', '').upper()}_SOURCE"
                if env_var not in env:
                    env[env_var] = os.path.join(metal_dir, f)

    try:
        cwd = os.path.dirname(os.path.abspath(__file__)) + "/.."
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=cwd,
        )
        if result.returncode != 0:
            return f"ERROR rc={result.returncode} stderr={result.stderr[:200]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except Exception as e:
        return f"ERROR: {e}"


def score_output(text, markers, inverse=False):
    """Score output by counting marker occurrences."""
    if not text or text.startswith("ERROR:"):
        return -9999.0
    score = 0
    for marker in markers:
        score += text.lower().count(marker.lower())
    if inverse:
        score = -score
    return score


def main():
    parser = argparse.ArgumentParser(description="Auto-tune steering vector scale")
    parser.add_argument("--model", default=os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf"))
    parser.add_argument("--vector", required=True, help="Path to steering vector .f32 file")
    parser.add_argument("--prompt", required=True, help="Test prompt")
    parser.add_argument("--scales", default="-3.0,-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0,3.0",
                        help="Comma-separated list of scales to try")
    parser.add_argument("--marker", action="append", default=[],
                        help="String markers to score on (can be used multiple times)")
    parser.add_argument("--inverse", action="store_true",
                        help="Minimize markers instead of maximizing")
    parser.add_argument("--n-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--backend", default="metal", choices=["metal", "cpu"])
    parser.add_argument("--output", help="Write results to file")
    args = parser.parse_args()

    if not args.marker:
        print("Warning: no markers specified, using length as score")

    scales = [float(s.strip()) for s in args.scales.split(",")]
    results = []

    print(f"Model: {args.model}")
    print(f"Vector: {args.vector}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Markers: {args.marker} ({'minimize' if args.inverse else 'maximize'})")
    print(f"Scales to try: {scales}")
    print("-" * 60)

    for scale in scales:
        print(f"  scale={scale:+.1f} ... ", end="", flush=True)
        text = run_ds4(
            args.model, args.vector, scale,
            args.prompt, args.n_tokens,
            args.temperature, args.backend,
        )
        score = score_output(text, args.marker, args.inverse)
        results.append((scale, score, text))
        print(f"score={score:.1f} text={text[:60]!r}")

    print("-" * 60)
    best = max(results, key=lambda x: x[1])
    print(f"BEST: scale={best[0]:+.1f} score={best[1]:.1f}")
    print(f"Output: {best[2]}")

    if args.output:
        with open(args.output, "w") as f:
            f.write("scale,score,text\n")
            for scale, score, text in results:
                f.write(f"{scale},{score},{text.replace(chr(10), ' ')}\n")
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()

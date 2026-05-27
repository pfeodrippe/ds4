#!/usr/bin/env python3
"""Sensorimotor tool loop for DS4.

Runs ds4 in a REPL-like loop where generated text is interpreted as actions,
executed in an external REPL, and the results are fed back as context.

Usage with mock REPL (for testing):
  python3 sensorimotor_loop.py --ds4 ./ds4 --model qwen3-coder.gguf --backend cpu \
    --prompt "You are in a Python REPL. Type expressions and observe results.\n\n>>> " \
    --mock-repl --iterations 3

Usage with Clojure REPL:
  python3 sensorimotor_loop.py --ds4 ./ds4 --model qwen3-coder.gguf --backend cpu \
    --prompt "You are in a Clojure REPL. Type expressions and observe results.\n\nuser=> " \
    --clojure-repl
"""

import argparse
import subprocess
import sys
import re
import os
import json


def run_ds4(ds4_path, model_path, backend, prompt, ctx_size=512, n_tokens=30, temp=0.0):
    """Run ds4 with a prompt and return generated text."""
    cmd = [
        ds4_path,
        "-m", model_path,
        "--" + backend,
        "--ctx", str(ctx_size),
        "-n", str(n_tokens),
        "--temp", str(temp),
        "-p", prompt,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"ds4 stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ds4 exited with code {result.returncode}")
    return result.stdout.strip()


def extract_code_block(text):
    """Extract the first code-like line from generated text."""
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        # Strip trailing assignment operators and results
        line = re.sub(r"\s*=\s*\d+.*$", "", line)
        line = line.strip()
        if not line:
            continue
        if line and not line.startswith("(") and not line.startswith("["):
            # Skip natural language, look for code
            if any(c in line for c in "+-*/()[]{}\""):
                return line
        elif line and (line.startswith("(") or line.startswith("[")):
            return line
    return lines[0].strip() if lines else ""


def mock_repl_eval(code):
    """Evaluate code in a mock Python-like REPL."""
    code = code.strip()
    if not code:
        return ""
    # Remove trailing equals sign if present
    if code.endswith("="):
        code = code[:-1].strip()
    # Simple arithmetic evaluation
    try:
        # Only allow safe expressions
        allowed = {"__builtins__": {}}
        result = eval(code, allowed, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def clojure_repl_eval(code, repl_proc):
    """Evaluate code in a Clojure REPL subprocess."""
    code = code.strip()
    if not code:
        return ""
    repl_proc.stdin.write(code + "\n")
    repl_proc.stdin.flush()
    # Read output until prompt appears
    output = ""
    while True:
        line = repl_proc.stdout.readline()
        if not line:
            break
        if "user=>" in line or "=>" in line:
            break
        output += line
    return output.strip()


def main():
    parser = argparse.ArgumentParser(description="DS4 Sensorimotor Tool Loop")
    parser.add_argument("--ds4", default="./ds4", help="Path to ds4 binary")
    parser.add_argument("--model", default="qwen3-coder.gguf", help="Model path")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "metal"], help="Backend")
    parser.add_argument("--prompt", required=True, help="Initial prompt")
    parser.add_argument("--ctx-size", type=int, default=512, help="Context size")
    parser.add_argument("--n-tokens", type=int, default=20, help="Tokens per generation")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--iterations", type=int, default=3, help="Loop iterations")
    parser.add_argument("--mock-repl", action="store_true", help="Use mock Python REPL")
    parser.add_argument("--clojure-repl", action="store_true", help="Use Clojure REPL")
    parser.add_argument("--output", help="Log results to file")
    args = parser.parse_args()

    if not args.mock_repl and not args.clojure_repl:
        print("Error: specify --mock-repl or --clojure-repl", file=sys.stderr)
        sys.exit(1)

    clojure_proc = None
    if args.clojure_repl:
        try:
            clojure_proc = subprocess.Popen(
                ["clojure"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Wait for initial prompt
            clojure_proc.stdout.readline()
        except FileNotFoundError:
            print("Error: clojure not found in PATH", file=sys.stderr)
            sys.exit(1)

    log = []
    prompt = args.prompt

    print(f"=== Sensorimotor Loop ({'mock' if args.mock_repl else 'clojure'} REPL) ===\n")

    for iteration in range(args.iterations):
        print(f"--- Iteration {iteration + 1} ---")
        print(f"Prompt: {prompt[:200]}...")

        try:
            response = run_ds4(
                args.ds4, args.model, args.backend,
                prompt, args.ctx_size, args.n_tokens, args.temp
            )
        except RuntimeError as e:
            print(f"Generation failed: {e}", file=sys.stderr)
            break

        print(f"Generated: {response[:200]}...")

        code = extract_code_block(response)
        print(f"Extracted: {code}")

        if args.mock_repl:
            result = mock_repl_eval(code)
        else:
            result = clojure_repl_eval(code, clojure_proc)

        print(f"Result: {result}\n")

        log.append({
            "iteration": iteration + 1,
            "prompt": prompt,
            "generated": response,
            "code": code,
            "result": result,
        })

        # Build next prompt with history
        prompt = prompt + response + "\n" + result + "\n>>> "

    if clojure_proc:
        clojure_proc.stdin.close()
        clojure_proc.wait()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log written to {args.output}")

    print("=== Loop complete ===")


if __name__ == "__main__":
    main()

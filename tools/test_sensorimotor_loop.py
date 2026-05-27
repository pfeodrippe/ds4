#!/usr/bin/env python3
"""End-to-end test for the sensorimotor loop.

Uses the mock REPL to verify the round-trip without external dependencies.
"""

import os
import subprocess
import sys
import tempfile


def test_mock_repl_arithmetic():
    """Test that the sensorimotor loop can evaluate arithmetic."""
    ds4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ds4")
    model = os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf")

    if not os.path.exists(ds4_path):
        print(f"SKIP: ds4 binary not found at {ds4_path}")
        return 0

    loop_path = os.path.join(os.path.dirname(__file__), "sensorimotor_loop.py")
    prompt = "You are a calculator. Only output the arithmetic expression, nothing else.\n\nCalculate 2+3: "

    cmd = [
        sys.executable, loop_path,
        "--ds4", ds4_path,
        "--model", model,
        "--backend", "cpu",
        "--prompt", prompt,
        "--mock-repl",
        "--iterations", "1",
        "--n-tokens", "5",
        "--ctx-size", "128",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"FAIL: sensorimotor loop exited with code {result.returncode}")
        return 1

    # Verify that some output was generated and evaluated
    if "Result:" not in result.stdout:
        print("FAIL: no result found in output")
        return 1

    print("PASS: sensorimotor loop round-trip works")
    return 0


if __name__ == "__main__":
    sys.exit(test_mock_repl_arithmetic())

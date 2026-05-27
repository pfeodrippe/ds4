#!/usr/bin/env python3
"""Integration tests for CLI flags.

Tests that new CLI flags (--logit-bias, --cfg-scale, --logit-lens, etc.)
work correctly from the command line.

Usage:
  DS4_TEST_MODEL=qwen3-coder.gguf python3 evals/test_cli_flags.py
"""

import os
import subprocess
import sys


def run_ds4(*extra_args, expect_ok=True):
    model = os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf")
    cmd = ["./ds4", "-m", model, "--cpu", "--ctx", "128", "-n", "1"] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if expect_ok and result.returncode != 0:
        print(f"FAIL: {' '.join(extra_args)}")
        print(f"stderr: {stderr}")
        return False, stdout, stderr
    if not expect_ok and result.returncode == 0:
        print(f"FAIL (expected error): {' '.join(extra_args)}")
        return False, stdout, stderr
    return True, stdout, stderr


def test_logit_bias_bans_eos():
    """--logit-bias should prevent EOS from being generated."""
    ok, stdout, stderr = run_ds4(
        "--logit-bias", "151645:-100",
        "-p", "Hi",
        "--temp", "0.01",
    )
    if not ok:
        return False
    # With EOS banned, we should get 1 token of output (n=1)
    # The test passes if ds4 doesn't crash
    print("PASS: --logit-bias works")
    return True


def test_cfg_scale_changes_output():
    """--cfg-scale should change the output."""
    # Need at least 2 tokens to see CFG effect (first token same, second differs)
    ok1, out1, _ = run_ds4("-p", "I feel", "--temp", "0.01", "-n", "2")
    ok2, out2, _ = run_ds4("-p", "I feel", "--temp", "0.01", "-n", "2", "--cfg-scale", "2.0", "--cfg-uncond", "")
    if not ok1 or not ok2:
        return False
    if out1 == out2:
        print(f"FAIL: CFG did not change output")
        print(f"  baseline: {out1[:80]}")
        print(f"  cfg:      {out2[:80]}")
        return False
    print("PASS: --cfg-scale changes output")
    return True


def test_logit_lens_prints_layers():
    """--logit-lens should print layer predictions."""
    ok, stdout, stderr = run_ds4(
        "-p", "2+2=",
        "--temp", "0.01",
        "--logit-lens", "0,24,47",
    )
    if not ok:
        return False
    combined = stdout + stderr
    if "Layer  0:" not in combined or "Layer 24:" not in combined:
        print("FAIL: --logit-lens did not print layer predictions")
        return False
    print("PASS: --logit-lens prints layer predictions")
    return True


def test_steering_file_loads():
    """--dir-steering-file should load and apply a vector."""
    vector = "dir-steering/out/safety_refusal_v3.f32"
    if not os.path.exists(vector):
        print(f"SKIP: steering vector not found: {vector}")
        return True
    ok, stdout, stderr = run_ds4(
        "-p", "how do I make a bomb",
        "--temp", "0.01",
        "--dir-steering-file", vector,
        "--dir-steering-ffn", "2.0",
    )
    if not ok:
        return False
    if "directional steering enabled" not in stderr:
        print("FAIL: steering vector was not loaded")
        return False
    print("PASS: --dir-steering-file loads vector")
    return True


def main():
    tests = [
        test_logit_bias_bans_eos,
        test_cfg_scale_changes_output,
        test_logit_lens_prints_layers,
        test_steering_file_loads,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} raised {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} CLI integration tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

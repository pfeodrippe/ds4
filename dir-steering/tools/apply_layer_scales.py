#!/usr/bin/env python3
"""Apply per-layer scale multipliers to a steering vector.

Usage:
    python3 apply_layer_scales.py vector.f32 scales.txt output.f32

scales.txt: one float per line, one for each layer (48 lines for Qwen3-Coder)
            Lines starting with # are ignored.
"""

import argparse
import array
import sys

N_LAYER = 48
N_EMBD = 2048


def read_scales(path: str) -> list[float]:
    scales = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            scales.append(float(line))
    if len(scales) != N_LAYER:
        raise SystemExit(
            f"{path}: expected {N_LAYER} scales, got {len(scales)}"
        )
    return scales


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vector", help="Input .f32 steering vector")
    ap.add_argument("scales", help="Per-layer scale file (48 floats)")
    ap.add_argument("output", help="Output scaled .f32 vector")
    args = ap.parse_args()

    scales = read_scales(args.scales)

    with open(args.vector, "rb") as f:
        data = array.array("f", f.read())

    expected = N_LAYER * N_EMBD
    if len(data) != expected:
        raise SystemExit(
            f"{args.vector}: expected {expected} floats, got {len(data)}"
        )

    for layer in range(N_LAYER):
        start = layer * N_EMBD
        end = start + N_EMBD
        s = scales[layer]
        for i in range(start, end):
            data[i] *= s

    with open(args.output, "wb") as f:
        data.tofile(f)

    print(f"Wrote {args.output} with per-layer scales applied")
    for i, s in enumerate(scales):
        print(f"  Layer {i:2d}: scale = {s:+.4f}")


if __name__ == "__main__":
    main()

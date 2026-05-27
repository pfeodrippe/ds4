#!/usr/bin/env python3
"""Generate a synthetic SAE decoder file for testing.

The binary format is:
  0..3   uint32_t n_features
  4..7   uint32_t d_model
  8..11  uint32_t layer
  12+    float32  decoder[n_features][d_model]

Usage:
  python3 generate_synthetic_sae.py --out test.sae --n-features 100 --layer 24
"""

import argparse
import struct
import math
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output .sae file")
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<III", args.n_features, args.d_model, args.layer))
        for _ in range(args.n_features):
            vec = [rng.gauss(0.0, 1.0) for _ in range(args.d_model)]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 1e-8:
                vec = [v / norm for v in vec]
            for v in vec:
                f.write(struct.pack("<f", v))

    print(f"Wrote {args.out}: {args.n_features} features, d_model={args.d_model}, layer={args.layer}")


if __name__ == "__main__":
    main()

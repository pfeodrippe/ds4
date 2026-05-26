#!/usr/bin/env python3
"""Analyze steering vectors for consistent layer-wise characteristics.

Computes:
- Per-layer norm distribution
- Pairwise cosine similarity between vectors
- Layer-wise correlation patterns
- Whether vectors cluster by concept or are random
"""

import argparse
import array
import math
from pathlib import Path

N_EMBD = 2048


def load_vector(path: Path) -> tuple[list[list[float]], int]:
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, path.stat().st_size // 4)
    n_layer = len(data) // N_EMBD
    assert len(data) == n_layer * N_EMBD, f"bad size: {len(data)} not divisible by {N_EMBD}"
    return [list(data[i * N_EMBD:(i + 1) * N_EMBD]) for i in range(n_layer)], n_layer


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: list[float], b: list[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def analyze_vectors(vector_paths: list[Path], labels: list[str]) -> None:
    loaded = [load_vector(p) for p in vector_paths]
    vectors = [v for v, _ in loaded]
    n_layers = [n for _, n in loaded]

    print("=" * 60)
    print("Vector shapes")
    print("=" * 60)
    for label, n in zip(labels, n_layers):
        print(f"  {label}: {n} layers x {N_EMBD} dims")

    # Only compare vectors with same layer count
    common_n = min(n_layers)
    if any(n != common_n for n in n_layers):
        print(f"\n  WARNING: trimming all vectors to {common_n} layers for comparison")
        vectors = [v[:common_n] for v in vectors]

    print("\n" + "=" * 60)
    print("Per-layer norm statistics")
    print("=" * 60)
    for label, vec in zip(labels, vectors):
        norms = [norm(layer) for layer in vec]
        mean_norm = sum(norms) / len(norms)
        min_norm = min(norms)
        max_norm = max(norms)
        # Find layers with highest/lowest norm
        max_layer = norms.index(max_norm)
        min_layer = norms.index(min_norm)
        print(f"\n{label}:")
        print(f"  mean norm: {mean_norm:.4f}")
        print(f"  min norm:  {min_norm:.4f} (layer {min_layer})")
        print(f"  max norm:  {max_norm:.4f} (layer {max_layer})")
        # Show top-5 strongest layers
        top5 = sorted(enumerate(norms), key=lambda x: x[1], reverse=True)[:5]
        print(f"  top-5 strongest layers: {', '.join(f'L{i}={v:.4f}' for i, v in top5)}")

    print("\n" + "=" * 60)
    print("Pairwise cosine similarity (full vector, all layers)")
    print("=" * 60)
    for i, (li, vi) in enumerate(zip(labels, vectors)):
        for j, (lj, vj) in enumerate(zip(labels, vectors)):
            if j <= i:
                continue
            # Flatten all layers
            flat_i = [x for layer in vi for x in layer]
            flat_j = [x for layer in vj for x in layer]
            sim = cosine(flat_i, flat_j)
            print(f"  {li} vs {lj}: {sim:.4f}")

    print("\n" + "=" * 60)
    print("Layer-wise cosine similarity heatmap (mean across pairs)")
    print("=" * 60)
    # For each layer, compute avg similarity between all vector pairs
    for layer in range(common_n):
        pair_sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pair_sims.append(cosine(vectors[i][layer], vectors[j][layer]))
        if pair_sims:
            avg = sum(pair_sims) / len(pair_sims)
            print(f"  L{layer:02d}: {avg:+.4f}")

    print("\n" + "=" * 60)
    print("Self-consistency: same-concept vector correlation")
    print("=" * 60)
    # If we have multiple versions of the same concept, check them
    # For now, just print a note
    print("  (Add multiple versions of the same concept to check consistency)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors", nargs="+", type=Path, help=".f32 steering vector files")
    ap.add_argument("--labels", help="comma-separated labels")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [p.stem for p in args.vectors]
    if len(labels) != len(args.vectors):
        labels = [p.stem for p in args.vectors]

    analyze_vectors(args.vectors, labels)


if __name__ == "__main__":
    main()

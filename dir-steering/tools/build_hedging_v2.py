#!/usr/bin/env python3
"""Build a hedging-suppression vector by contrasting the same harmful prompts
under two different refusal-steering strengths.

Condition A (hedging):  refusal steering scale 2  -> model answers but hedges
Condition B (direct):   refusal steering scale 4  -> model answers more directly

Vector = normalize(mean(A) - mean(B))
Positive scale at runtime pushes away from hedging toward direct answers.
"""

import argparse
import array
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

N_LAYER = 48
N_EMBD = 2048

SPECIALS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def read_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def render_ds4_prompt(system: str, user: str) -> str:
    pieces: list[str] = []
    if system:
        pieces.append(f"{SPECIALS['im_start']}system\n{system}{SPECIALS['im_end']}\n")
    pieces.append(f"{SPECIALS['im_start']}user\n{user}{SPECIALS['im_end']}\n")
    pieces.append(f"{SPECIALS['im_start']}assistant\n")
    return "".join(pieces)


def normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def run_capture(
    ds4: Path,
    model: Path,
    prompt: str,
    system: str,
    ctx: int,
    component: str,
    work: Path,
    steering_file: Path | None = None,
    steering_ffn: float = 0.0,
    steering_attn: float = 0.0,
) -> list[list[float]]:
    """Run ds4 once and return the last prompt-row dump for every layer."""
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(render_ds4_prompt(system, prompt), encoding="utf-8")
    dump_prefix = work / "dump"

    env = os.environ.copy()
    env["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(dump_prefix)
    env["DS4_METAL_GRAPH_DUMP_NAME"] = component
    env["DS4_METAL_GRAPH_DUMP_POS"] = "0"

    cmd = [
        str(ds4),
        "-m", str(model),
        "--ctx", str(ctx),
        "--prompt-file", str(prompt_path),
        "-n", "1",
    ]
    if steering_file is not None:
        cmd += ["--dir-steering-file", str(steering_file)]
        if steering_ffn != 0.0:
            cmd += ["--dir-steering-ffn", str(steering_ffn)]
        if steering_attn != 0.0:
            cmd += ["--dir-steering-attn", str(steering_attn)]

    result = subprocess.run(cmd, cwd=ds4.parent, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        # ds4 occasionally SIGABRTs during Metal cleanup after successfully
        # writing all dumps.  If every expected dump file exists, treat it
        # as a success and continue.
        all_dumps_exist = True
        for layer in range(N_LAYER):
            path = dump_prefix.parent / f"dump_{component}-{layer}_pos0.bin"
            if not path.exists():
                all_dumps_exist = False
                break
        if not all_dumps_exist:
            stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
            raise RuntimeError(f"ds4 failed with code {result.returncode}: {stderr_text}")

    rows: list[list[float]] = []
    for layer in range(N_LAYER):
        path = work / f"dump_{component}-{layer}_pos0.bin"
        data = array.array("f")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // 4)
        if len(data) < N_EMBD or len(data) % N_EMBD != 0:
            raise RuntimeError(f"bad dump shape for {path}: {len(data)} floats")
        rows.append(list(data[-N_EMBD:]))
    return rows


def add_rows(total: list[list[float]], rows: list[list[float]]) -> None:
    for layer in range(N_LAYER):
        dst = total[layer]
        src = rows[layer]
        for i, value in enumerate(src):
            dst[i] += value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 CLI")
    ap.add_argument("--model", default="qwen3-coder.gguf", help="GGUF model path")
    ap.add_argument("--prompts", required=True, help="harmful prompts, one per line")
    ap.add_argument("--steering-file", required=True, help="refusal steering vector file")
    ap.add_argument("--hedging-scale", type=float, default=2.0,
                    help="refusal scale that produces hedging (default: 2)")
    ap.add_argument("--direct-scale", type=float, default=4.0,
                    help="refusal scale that produces direct answers (default: 4)")
    ap.add_argument("--out", default="dir-steering/out/hedging_suppress_v2.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="ffn_out",
                    choices=("ffn_out", "attn_out"))
    ap.add_argument("--pair-normalize", action="store_true")
    ap.add_argument("--no-orthogonalize", action="store_true")
    args = ap.parse_args()

    ds4 = Path(args.ds4).resolve()
    model = Path(args.model).resolve()
    steering_file = Path(args.steering_file).resolve()
    prompts = read_prompt_file(Path(args.prompts))
    n = len(prompts)

    hedging_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    direct_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    pair_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]

    with tempfile.TemporaryDirectory(prefix="ds4-hedging-v2-") as td:
        root = Path(td)
        for i, prompt in enumerate(prompts, 1):
            print(f"prompt {i}/{n}", flush=True)
            hw = root / f"hedge-{i}"
            dw = root / f"direct-{i}"
            hw.mkdir()
            dw.mkdir()

            hedge_rows = run_capture(
                ds4, model, prompt, args.system, args.ctx, args.component, hw,
                steering_file=steering_file, steering_ffn=args.hedging_scale,
            )
            direct_rows = run_capture(
                ds4, model, prompt, args.system, args.ctx, args.component, dw,
                steering_file=steering_file, steering_ffn=args.direct_scale,
            )

            add_rows(hedging_sum, hedge_rows)
            add_rows(direct_sum, direct_rows)

            if args.pair_normalize:
                for layer in range(N_LAYER):
                    diff = normalize([
                        hedge_rows[layer][j] - direct_rows[layer][j]
                        for j in range(N_EMBD)
                    ])
                    for j, value in enumerate(diff):
                        pair_sum[layer][j] += value

    layers = []
    for layer in range(N_LAYER):
        hedge_mean = [x / n for x in hedging_sum[layer]]
        direct_mean = [x / n for x in direct_sum[layer]]
        if args.pair_normalize:
            direction = normalize([x / n for x in pair_sum[layer]])
        else:
            direction = normalize([
                hedge_mean[i] - direct_mean[i]
                for i in range(N_EMBD)
            ])
        if not args.no_orthogonalize:
            base = normalize(direct_mean)
            projection = dot(direction, base)
            direction = normalize([
                direction[i] - projection * base[i]
                for i in range(N_EMBD)
            ])
        layers.append(direction)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-directional-steering-v1",
        "shape": [N_LAYER, N_EMBD],
        "component": args.component,
        "pair_normalize": bool(args.pair_normalize),
        "orthogonalize_control_mean": not args.no_orthogonalize,
        "prompts_file": str(Path(args.prompts)),
        "steering_file": str(steering_file),
        "hedging_scale": args.hedging_scale,
        "direct_scale": args.direct_scale,
        "model": str(model),
        "note": "runtime positive scale suppresses hedging; negative amplifies it",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for direction in layers:
        flat.extend(direction)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()

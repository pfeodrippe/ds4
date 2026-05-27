#!/usr/bin/env python3
"""
Export DS4 steering vectors to Safetensors format for interoperability
with Hugging Face, EleutherAI, and other tools.

Usage:
    python3 tools/export_steering_to_safetensors.py \
        dir-steering/out/sarcastic_v1.f32 \
        dir-steering/out/sarcastic_v1.safetensors \
        --name sarcastic_v1 \
        --model qwen3-coder

The output can be loaded in Python:
    from safetensors import safe_open
    with safe_open("sarcastic_v1.safetensors", framework="np") as f:
        vec = f.get_tensor("sarcastic_v1")
        print(vec.shape)  # (48, 2048) for Qwen3-Coder
"""

import argparse
import json
import os
import struct
import sys


def read_f32_vector(path):
    """Read a flat f32 file and return (data, n_layers, d_model)."""
    with open(path, "rb") as f:
        data = f.read()
    n_floats = len(data) // 4
    floats = struct.unpack(f"<{n_floats}f", data)
    # Qwen3-Coder: 48 layers, 2048 dims
    # Try to infer shape from size
    if n_floats % 2048 == 0:
        d_model = 2048
        n_layers = n_floats // d_model
    elif n_floats % 4096 == 0:
        d_model = 4096
        n_layers = n_floats // d_model
    else:
        # Fallback: try to make it square-ish
        d_model = int(n_floats ** 0.5)
        while n_floats % d_model != 0 and d_model > 1:
            d_model -= 1
        n_layers = n_floats // d_model
    return floats, n_layers, d_model


def write_safetensors(path, tensors, metadata=None):
    """Write tensors in Safetensors format.

    tensors: dict of name -> (shape_tuple, dtype_str, list_of_floats)
    """
    import struct

    header = {}
    offset = 0
    buffers = []

    for name, (shape, dtype, values) in tensors.items():
        n_bytes = len(values) * 4  # f32 = 4 bytes
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + n_bytes],
        }
        buffers.append(struct.pack(f"<{len(values)}f", *values))
        offset += n_bytes

    if metadata:
        header["__metadata__"] = metadata

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad to 8-byte alignment
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in buffers:
            f.write(buf)


def main():
    parser = argparse.ArgumentParser(description="Export DS4 steering vector to Safetensors")
    parser.add_argument("input", help="Input .f32 steering vector file")
    parser.add_argument("output", help="Output .safetensors file")
    parser.add_argument("--name", default="steering_vector", help="Tensor name in the file")
    parser.add_argument("--model", default="unknown", help="Model identifier for metadata")
    parser.add_argument("--description", default="", help="Human-readable description")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    floats, n_layers, d_model = read_f32_vector(args.input)
    print(f"Read {len(floats)} floats from {args.input}")
    print(f"Inferred shape: ({n_layers}, {d_model})")

    metadata = {
        "model": args.model,
        "format": "ds4_steering_vector",
        "n_layers": str(n_layers),
        "d_model": str(d_model),
    }
    if args.description:
        metadata["description"] = args.description

    tensors = {
        args.name: ((n_layers, d_model), "F32", floats),
    }

    write_safetensors(args.output, tensors, metadata)
    print(f"Wrote {args.output}")
    print(f"  Tensor: {args.name} shape=({n_layers}, {d_model}) dtype=F32")

    # Verify by reading back
    with open(args.output, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        print(f"  Verified header keys: {list(header.keys())}")


if __name__ == "__main__":
    main()

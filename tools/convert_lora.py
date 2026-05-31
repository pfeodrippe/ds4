#!/usr/bin/env python3
"""
Convert MLX/PEFT LoRA adapters to DS4 binary format.

Usage:
    python3 tools/convert_lora.py \
        --input /tmp/lora_output/adapters.safetensors \
        --output /tmp/lora_output/ds4_lora.bin

DS4 binary format:
    Header (256 bytes):
        - magic: "DS4LORA" (8 bytes)
        - version: uint32 (1)
        - rank: uint32
        - alpha: float32
        - n_layers: uint32
        - reserved: 232 bytes
    
    Layer table (n_layers * 16 bytes each):
        - layer_idx: uint32
        - target: uint32 (0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down)
        - input_dim: uint32
        - output_dim: uint32
    
    Data (FP32):
        - For each layer: A matrix (rank * input_dim floats)
        - For each layer: B matrix (output_dim * rank floats)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required. Install with: pip install safetensors")
    sys.exit(1)


def parse_layer_key(key: str):
    """Parse MLX LoRA key into (layer_idx, target, matrix_type).
    
    Handles both standard and Qwen3-Coder MoE naming:
      - model.layers.{idx}.self_attn.{q|k|v|o}_proj.lora_a
      - model.layers.{idx}.mlp.gate.lora_a
      - model.layers.{idx}.mlp.switch_mlp.{gate|down}_proj.lora_a
    """
    parts = key.split('.')
    
    # Find layer index
    layer_idx = None
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            layer_idx = int(parts[i + 1])
            break
    
    # Determine target (skip MoE switch_mlp layers, only handle attention)
    target = None
    if 'switch_mlp' in parts:
        # MoE routed layers - skip for now
        return None, None, None
    elif 'self_attn' in parts:
        if 'q_proj' in parts:
            target = 0
        elif 'k_proj' in parts:
            target = 1
        elif 'v_proj' in parts:
            target = 2
        elif 'o_proj' in parts:
            target = 3
    elif 'mlp' in parts:
        return None, None, None
    
    # Determine matrix type
    matrix_type = None
    if (key.endswith('.lora_a') or key.endswith('.lora_A') or
            key.endswith('.lora_a.weight') or key.endswith('.lora_A.weight')):
        matrix_type = 'A'
    elif (key.endswith('.lora_b') or key.endswith('.lora_B') or
            key.endswith('.lora_b.weight') or key.endswith('.lora_B.weight')):
        matrix_type = 'B'
    
    return layer_idx, target, matrix_type


def convert(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Reading: {input_path}")
    
    with safe_open(input_path, framework="numpy") as f:
        keys = list(f.keys())
        
        # Collect layer info
        layers = {}  # (layer_idx, target) -> {'A': tensor, 'B': tensor}
        rank = None
        
        for key in keys:
            layer_idx, target, matrix_type = parse_layer_key(key)
            
            if layer_idx is None or target is None or matrix_type is None:
                continue  # Skip silently (MoE layers, etc.)
            
            tensor = f.get_tensor(key)
            
            # Skip 3D tensors (MoE expert weights)
            if len(tensor.shape) != 2:
                print(f"  Skipping {key}: shape {tensor.shape} (not 2D)")
                continue
            
            k = (layer_idx, target)
            if k not in layers:
                layers[k] = {}
            layers[k][matrix_type] = tensor
            
            if matrix_type == 'A':
                # MLX format: A is (input_dim, rank), B is (rank, output_dim)
                # Smaller dimension is rank
                if tensor.shape[0] < tensor.shape[1]:
                    detected_rank = tensor.shape[0]
                else:
                    detected_rank = tensor.shape[1]
                if rank is None:
                    rank = detected_rank
                elif rank != detected_rank:
                    print(f"  Warning: inconsistent rank {detected_rank} vs {rank} for {key}")
        
        if not layers:
            print("ERROR: No valid LoRA layers found!")
            sys.exit(1)
        
        # Determine alpha from MLX's saved config. MLX stores scale directly,
        # while DS4 stores alpha and applies alpha / rank at inference.
        alpha = rank * 2
        adapter_config = input_path.parent / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config, "r") as fcfg:
                cfg = json.load(fcfg)
            params = cfg.get("lora_parameters") or {}
            scale = params.get("scale")
            cfg_rank = params.get("rank")
            if cfg_rank is not None and int(cfg_rank) != rank:
                print(f"  Warning: adapter_config rank {cfg_rank} != tensor rank {rank}")
            if scale is not None:
                alpha = float(scale) * rank
        
        # Normalize shapes and keep only complete adapter pairs.
        normalized = {}
        for k in sorted(layers.keys()):
            layer = layers[k]
            if 'A' not in layer or 'B' not in layer:
                print(f"  Skipping layer {k}: missing A or B")
                continue

            a = layer['A']
            b = layer['B']

            # MLX/PEFT usually stores A as (rank, input_dim) or (input_dim, rank).
            if a.shape[0] != rank and a.shape[1] == rank:
                a = a.T
            if a.shape[0] != rank:
                raise ValueError(f"A matrix for {k} has shape {a.shape}; rank {rank} is not the first dimension")

            # MLX/PEFT usually stores B as (output_dim, rank) or (rank, output_dim).
            if b.shape[1] != rank and b.shape[0] == rank:
                b = b.T
            if b.shape[1] != rank:
                raise ValueError(f"B matrix for {k} has shape {b.shape}; rank {rank} is not the second dimension")

            normalized[k] = {
                'A': a.astype('float32'),
                'B': b.astype('float32'),
                'input_dim': int(a.shape[1]),
                'output_dim': int(b.shape[0]),
            }

        if not normalized:
            print("ERROR: No complete LoRA adapter pairs found!")
            sys.exit(1)

        # Sort layers
        sorted_layers = sorted(normalized.keys())
        n_layers = len(sorted_layers)
        
        print(f"Found {n_layers} LoRA layers")
        print(f"Rank: {rank}, Alpha: {alpha}")
        
        # Write binary
        print(f"Writing: {output_path}")
        with open(output_path, 'wb') as out:
            # Header (256 bytes)
            out.write(b'DS4LORA\x00')
            out.write(struct.pack('<I', 1))
            out.write(struct.pack('<I', rank))
            out.write(struct.pack('<f', float(alpha)))
            out.write(struct.pack('<I', n_layers))
            out.write(b'\x00' * 232)
            
            # Layer table.
            for layer_idx, target in sorted_layers:
                layer = normalized[(layer_idx, target)]
                out.write(struct.pack('<I', layer_idx))
                out.write(struct.pack('<I', target))
                out.write(struct.pack('<I', layer['input_dim']))
                out.write(struct.pack('<I', layer['output_dim']))
            
            # Data: A then B for each layer
            for layer_idx, target in sorted_layers:
                layer = normalized[(layer_idx, target)]
                out.write(layer['A'].tobytes())
                out.write(layer['B'].tobytes())
        
        # Verify
        file_size = output_path.stat().st_size
        print(f"File size: {file_size:,} bytes")
        
        print(f"\nConversion complete!")
        print(f"Use in DS4: ds4_lora_load(engine, \"{output_path}\")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input safetensors file')
    parser.add_argument('--output', required=True, help='Output DS4 binary file')
    args = parser.parse_args()
    
    convert(args.input, args.output)

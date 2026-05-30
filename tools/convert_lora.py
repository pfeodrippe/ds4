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
        - d_model: uint32
        - reserved: uint32
    
    Data (FP32):
        - For each layer: A matrix (rank * input_dim floats)
        - For each layer: B matrix (output_dim * rank floats)
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required. Install with: pip install safetensors")
    sys.exit(1)


def parse_layer_key(key: str):
    """Parse MLX LoRA key like 'model.layers.0.self_attn.q_proj.lora_a' 
    into (layer_idx, target, matrix_type)."""
    parts = key.split('.')
    
    # Find layer index
    layer_idx = None
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            layer_idx = int(parts[i + 1])
            break
    
    # Determine target
    target = None
    if 'q_proj' in parts:
        target = 0
    elif 'k_proj' in parts:
        target = 1
    elif 'v_proj' in parts:
        target = 2
    elif 'o_proj' in parts:
        target = 3
    elif 'gate_proj' in parts:
        target = 4
    elif 'up_proj' in parts:
        target = 5
    elif 'down_proj' in parts:
        target = 6
    
    # Determine matrix type
    matrix_type = None
    if key.endswith('.lora_a') or key.endswith('.lora_A') or key.endswith('.lora_A.weight'):
        matrix_type = 'A'
    elif key.endswith('.lora_b') or key.endswith('.lora_B') or key.endswith('.lora_B.weight'):
        matrix_type = 'B'
    
    return layer_idx, target, matrix_type


def convert(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Reading: {input_path}")
    
    with safe_open(input_path, framework="numpy") as f:
        keys = list(f.keys())
        
        # Collect layer info
        layers = {}  # (layer_idx, target) -> {'A': tensor, 'B': tensor, 'd_model': int}
        rank = None
        
        for key in keys:
            layer_idx, target, matrix_type = parse_layer_key(key)
            
            if layer_idx is None or target is None or matrix_type is None:
                print(f"  Skipping: {key}")
                continue
            
            tensor = f.get_tensor(key)
            
            k = (layer_idx, target)
            if k not in layers:
                layers[k] = {}
            layers[k][matrix_type] = tensor
            
            if matrix_type == 'A':
                # Detect orientation: whichever dimension is smaller is likely rank
                if tensor.shape[0] < tensor.shape[1]:
                    detected_rank = tensor.shape[0]
                    detected_d_model = tensor.shape[1]
                else:
                    detected_rank = tensor.shape[1]
                    detected_d_model = tensor.shape[0]
                if rank is None:
                    rank = detected_rank
                layers[k]['d_model'] = detected_d_model
        
        if not layers:
            print("ERROR: No valid LoRA layers found!")
            sys.exit(1)
        
        # Determine alpha from key names or use default
        alpha = rank * 2
        
        # Sort layers
        sorted_layers = sorted(layers.keys())
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
            
            # Layer table (with per-layer d_model)
            for layer_idx, target in sorted_layers:
                layer = layers[(layer_idx, target)]
                layer_d_model = layer['d_model']
                out.write(struct.pack('<I', layer_idx))
                out.write(struct.pack('<I', target))
                out.write(struct.pack('<I', layer_d_model))
                out.write(struct.pack('<I', 0))
            
            # Data: A then B for each layer
            for layer_idx, target in sorted_layers:
                layer = layers[(layer_idx, target)]
                a = layer['A']
                b = layer['B']
                layer_d_model = layer['d_model']
                
                    # A matrix: store as (rank, input_dim) floats
                if a.shape[0] != rank:
                    a = a.T
                assert a.shape[0] == rank, f"A rank mismatch: {a.shape[0]} vs {rank}"
                out.write(a.astype('float32').tobytes())
                
                # B matrix: store as (output_dim, rank) floats
                if b.shape[1] != rank:
                    b = b.T
                assert b.shape[1] == rank, f"B rank mismatch: {b.shape[1]} vs {rank}"
                out.write(b.astype('float32').tobytes())
        
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

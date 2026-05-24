# Qwen3-Coder Native

This repository keeps the DS4 native-engine shape, but the model target is now:

```text
Qwen3-Coder-30B-A3B-Instruct
```

The goal is a narrow, self-contained local inference engine for this one model:
C core, mmap-backed GGUF loading, strict fixed-shape validation, CLI/server
session APIs, and Metal-oriented acceleration. It is not a generic GGUF runner
and it is not an Ollama wrapper.

## Status

The in-place conversion now runs the Qwen3-Coder Q4_K_M GGUF through a native
CPU reference graph:

- `download_model.sh` downloads only `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`.
- `qwen3-coder.gguf` is the default model symlink.
- `ds4.c` validates `general.architecture=qwen3moe` and the fixed
  Qwen3-Coder-30B-A3B-Instruct shape.
- `ds4.c` binds Qwen tensor names such as `blk.N.attn_q.weight`,
  `blk.N.attn_k.weight`, `blk.N.attn_v.weight`,
  `blk.N.ffn_gate_exps.weight`, and related MoE tensors.
- Chat rendering uses Qwen ChatML.
- Generation uses Qwen grouped-query attention, q/k RMSNorm, RoPE theta
  10,000,000, normalized top-8 MoE routing, and Q4_K/Q6_K GGUF matvec kernels.

The default backend is CPU while the Metal graph is rewritten for Qwen tensor
shapes. The CPU path is correct-first and intentionally narrow; it is slow
because it recomputes the prompt for each generated token.

## Model

Default GGUF:

```text
unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

Validated fixed shape:

| Field | Value |
| --- | ---: |
| Architecture | `qwen3moe` |
| Layers | 48 |
| Embedding length | 2048 |
| Vocabulary | 151936 |
| Attention heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| Experts | 128 |
| Active experts | 8 |
| Dense FFN length | 5472 |
| Expert FFN length | 768 |
| Context | 262144 |

## Build

```sh
make ds4
```

## Download

```sh
./download_model.sh
```

The script stores the GGUF under `./gguf/` and updates:

```text
./qwen3-coder.gguf
```

## Inspect

```sh
./ds4 --inspect
```

This validates the model metadata and tensor binding without initializing the
Metal graph.

## Generate

```sh
./ds4 -p "hello" -n 1 --ctx 64 --backend cpu
```

The first native smoke test with the downloaded GGUF generated:

```text
Hello
```

## Prompt Format

The renderer uses Qwen ChatML:

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

## Porting Checklist

- Replace DS4 compressed-attention CPU path with Qwen3 GQA. Done for the CPU
  reference graph.
- Replace DS4 mHC residual flow with Qwen RMSNorm residual blocks. Done for the
  CPU reference graph.
- Implement Qwen MoE router top-k and expert execution. Done for Q4_K/Q6_K
  routed experts.
- Dispatch Q4_K_M GGUF tensor types in matvec/matmul kernels. Done for the CPU
  reference graph.
- Replace full-prompt recompute with an ordinary per-layer K/V cache.
- Rewrite Metal kernels around Qwen tensor shapes.
- Tighten server disk-cache semantics around Qwen's standard KV layout.

# Qwen3-Coder Implementation Plan

This file records the narrow model decision and the in-place porting work for
the DS4 codebase.

## Target

The engine supports exactly one runtime target:

```text
Qwen3-Coder-30B-A3B-Instruct
```

Default local GGUF:

```text
unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

The repository keeps the DS4 shape: one model-specific native engine, mmap GGUF
loading, CLI/server/session APIs, and no generic GGUF runtime layer.

## Research Sources

- Official model repository:
  https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Runtime GGUF repository:
  https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
- Qwen3 MoE Transformers implementation:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py

The GGUF metadata and tensor directory are validated locally by `./ds4 --inspect`.

## Fixed Shape

| Field | Value |
| --- | ---: |
| GGUF architecture | `qwen3moe` |
| Layers | 48 |
| Hidden size | 2048 |
| Vocabulary | 151936 |
| Query heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| RoPE theta | 10000000 |
| Context length | 262144 |
| Experts | 128 |
| Routed experts per token | 8 |
| Dense FFN length | 5472 |
| Expert FFN length | 768 |

The downloaded Q4_K_M GGUF contains 579 tensors:

- F32: 241 tensors
- Q4_K: 289 tensors
- Q6_K: 49 tensors

## Implemented

- `download_model.sh` downloads only the selected Qwen3-Coder GGUF.
- Default model path is `qwen3-coder.gguf`.
- `ds4.c` validates Qwen metadata and binds Qwen tensor names.
- Chat rendering uses ChatML: `<|im_start|>role\n...<|im_end|>`.
- Token embeddings can expand Q4_K rows from the mmaped GGUF.
- Dense projections dispatch F32, F16, Q8_0, Q4_K, and Q6_K.
- Routed expert matvec dispatch handles the selected Q4_K/Q6_K expert tensors.
- Native CPU generation now executes:
  - RMSNorm residual blocks
  - grouped-query causal attention
  - per-head q/k RMSNorm
  - full 128-dimensional RoPE with theta 10000000
  - normalized top-8 MoE routing
  - final RMSNorm and output projection

## Verification

Commands run against the real downloaded GGUF:

```sh
make ds4 ds4-server ds4-agent
./ds4 --inspect -m qwen3-coder.gguf
make ds4_test ds4-eval
./ds4-eval --self-test-extractors
./ds4_test --server
./ds4 -m qwen3-coder.gguf -p 'hi' -n 1 --ctx 64 --backend cpu --nothink
```

The one-token smoke generated `Hello`.

## Remaining Engineering Work

The current CPU graph is a correctness-first reference path. It recomputes the
full prompt for each generated token, so it is slow but model-native.

Next work should stay in the existing DS4 files:

- Replace full-prompt recompute with a standard per-layer Qwen K/V cache.
- Rewrite the Metal graph and Metal kernels for Qwen tensor shapes.
- Adjust server disk session payloads from DS4 compressed KV state to Qwen
  standard KV state.
- Add a deterministic logits comparison against a known Qwen reference run when
  an official or locally generated fixture is available.

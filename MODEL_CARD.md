# Qwen3-Coder-30B-A3B-Instruct Model Card Synopsis

This project targets exactly one model:

```text
Qwen3-Coder-30B-A3B-Instruct
```

Primary source:

- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct

Runtime GGUF source:

- https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

## Model Family

Qwen3-Coder-30B-A3B-Instruct is a Qwen3 MoE coding model. It has about 30B total
parameters and about 3B active parameters per token, making it a practical
target for a 48 GiB Apple Silicon machine when quantized.

The repository default GGUF is:

```text
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

## Fixed Runtime Shape

The native loader validates the Qwen3 MoE GGUF metadata against this fixed
shape:

| Field | Value |
| --- | ---: |
| Architecture | `qwen3moe` |
| Layers | 48 |
| Embedding length | 2048 |
| Vocabulary | 151936 |
| Attention heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| Routed experts | 128 |
| Active experts | 8 |
| Dense FFN length | 5472 |
| Expert FFN length | 768 |
| Training context | 262144 tokens |

## Prompt Format

The chat renderer uses Qwen's ChatML-style prompt:

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

The previous DeepSeek DSML and thinking-mode prompt controls are not part of the
target model contract.

## Implementation Direction

The codebase keeps the original DS4 structure: a narrow C model-specific
runtime, mmap-backed GGUF loading, CLI/server/session APIs, and Metal-oriented
accelerator files. Internals are being rewritten in place for Qwen3-Coder:

- fixed metadata validation now targets `qwen3moe`
- tensor binding now targets Qwen attention and MoE tensor names
- chat rendering now targets Qwen ChatML
- download defaults now target the single Q4_K_M Qwen GGUF
- native CPU generation now uses Qwen grouped-query attention, per-head q/k
  RMSNorm, full-head RoPE with theta 10,000,000, normalized top-8 MoE routing,
  and Q4_K/Q6_K GGUF tensor kernels

Remaining work is performance work: replace full-prompt CPU recompute with a
standard Qwen KV cache and adapt the Metal graph to this tensor layout.

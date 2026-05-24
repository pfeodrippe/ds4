# Agent Notes

`ds4.c` is being rewritten in place as a Qwen3-Coder-30B-A3B-Instruct specific
native inference engine. It remains deliberately narrow: not a generic GGUF
runner, not an Ollama wrapper, and not a multi-model framework.

## Goals

- Keep the original DS4 project shape: C core, Objective-C only for Metal,
  kernels under `metal/`, narrow public API in `ds4.h`, CLI/server outside
  tensor internals.
- Support exactly one model: `Qwen3-Coder-30B-A3B-Instruct`.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Preserve strict GGUF metadata and tensor validation.
- Prefer correctness before speed; add Metal acceleration only after the CPU
  graph semantics are clear.

## Current Conversion State

- Download target: Qwen3-Coder Q4_K_M GGUF.
- Default model path: `qwen3-coder.gguf`.
- Metadata validation: `qwen3moe` fixed shape.
- Tensor binding: Qwen attention and MoE tensor names.
- Chat rendering: Qwen ChatML.
- Native CPU reference generation: Qwen GQA, q/k RMSNorm, RoPE, top-8 MoE, and
  Q4_K/Q6_K matvec kernels.
- Metal graph: still needs a Qwen rewrite; CPU is the default backend until
  that graph is Qwen-shaped.

## Testing

Use `make ds4` for build validation. Use `./ds4 --inspect -m qwen3-coder.gguf`
after downloading the model to validate the real GGUF. Use a tiny CPU smoke such
as `./ds4 -p "hi" -n 1 --ctx 64 --backend cpu` to verify generation.

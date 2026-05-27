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
Qwen graph:

- `download_model.sh` downloads only `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`.
- `qwen3-coder.gguf` is the default model symlink.
- `ds4.c` validates `general.architecture=qwen3moe` and the fixed
  Qwen3-Coder-30B-A3B-Instruct shape.
- `ds4.c` binds Qwen tensor names such as `blk.N.attn_q.weight`,
  `blk.N.attn_k.weight`, `blk.N.attn_v.weight`,
  `blk.N.ffn_gate_exps.weight`, and related MoE tensors.
- Chat rendering uses Qwen ChatML.
- CPU and Metal generation use Qwen grouped-query attention, q/k RMSNorm, RoPE theta
  10,000,000, normalized top-8 MoE routing, and Q4_K/Q6_K GGUF matvec kernels.
- The Metal backend is the default on macOS and uses Qwen-specific K/V caches,
  attention, router, MoE, output projection, and directional steering.

The CPU backend remains available as a reference/debug backend. It is not the
macOS default.

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
./ds4 -p "hello" -n 3 --ctx 64 --backend metal
```

The native Metal smoke test with the downloaded GGUF generated:

```text
Hello! How
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

## Clojure FFI Bindings

A Clojure project in `clj-ds4/` provides REPL-friendly Panama FFI bindings:

```bash
make libds4.dylib
make clj-test          # run Clojure integration tests
cd clj-ds4 && clojure -M:repl
```

```clojure
(require '[ds4-clj.core :as ds4])
(def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
(def session (ds4/create-session engine 512))

;; Generation, logit lens, CFG, steering, SAE — all from the REPL
(ds4/generate engine session "The capital of France is" {:n-tokens 10 :temperature 0.0})
(ds4/logit-lens engine session "The capital of France is" [0 12 24 35 47])
```

See `clj-ds4/src/ds4_clj/examples.clj` for interactive comment-block examples.

## Research Features

| Feature | CLI / API | Clojure | Description |
|---------|-----------|---------|-------------|
| Directional steering | `--dir-steering-file` | `open-engine` with `:steering-file` | Sarcastic, refusal, funny, violent vectors |
| Per-layer scales | `apply_layer_scales.py` | — | Offline layer-wise scale baking |
| Logit bias / token ban | `--logit-bias` | `set-logit-bias` | Force/ban tokens at sampling |
| Classifier-Free Guidance | `--cfg-scale` | `enable-cfg` | Conditional vs unconditional logits |
| Logit lens | `--logit-lens` | `logit-lens` | Per-layer top-k predictions (Metal + CPU) |
| SAE steering | `ds4_engine_load_sae` | `load-sae`, `sae-steer-multi` | Multi-feature sparse autoencoder steering |
| Sensorimotor loop | `sensorimotor_loop.py` | — | Model types into REPL, observes results |
| Steering auto-tuning | `tune_steering.py` | — | Grid-search optimal scale per prompt |
| Safetensors export | `export_steering_to_safetensors.py` | — | Share vectors in HF format |

## Porting Checklist

- Replace DS4 compressed-attention path with Qwen3 GQA. Done for CPU and Metal.
- Replace DS4 mHC residual flow with Qwen RMSNorm residual blocks. Done for the
  CPU and Metal graphs.
- Implement Qwen MoE router top-k and expert execution. Done for Q4_K/Q6_K
  routed experts.
- Dispatch Q4_K_M GGUF tensor types in matvec/matmul kernels. Done for CPU and
  Metal.
- Replace full-prompt recompute with an ordinary per-layer K/V cache. Done for
  CPU and Metal.
- Rewrite Metal kernels around Qwen tensor shapes. Done for token-major
  generation and sessions.
- Tighten server disk-cache semantics around Qwen's standard KV layout.

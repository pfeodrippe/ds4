# Qwen3 50+ tok/s Performance Implementation

Date: 2026-06-01

## Target

Run `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` with the custom DS4 Metal inference engine at or above 50 generated tokens/s on the local 48 GB M3 Max machine, without switching to Ollama or a CPU fallback.

## References Checked

- Qwen model card: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- llama.cpp Qwen3MoE model graph: https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3moe.cpp
- llama.cpp MoE graph builder: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp
- llama.cpp Metal kernels: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal
- Apple Metal compute encoders: https://developer.apple.com/documentation/metal/mtlcomputecommandencoder

## Baseline

Command:

```sh
./ds4-bench --prompt-file /tmp/ds4_long_numbers_200_prompt.txt \
  -m qwen3-coder.gguf --backend metal \
  --ctx-start 14 --ctx-max 14 --ctx-alloc 512 --gen-tokens 80
```

Baseline result before the accepted fix:

```text
14,14,54.11,80,40.43,952300
```

Longer forced-decode baseline:

```text
14,14,55.15,300,37.21,952300
```

Ollama comparison measured on the same model family/quantization:

```text
eval rate: 80.61 tokens/s
```

## Profiling Result

Low-overhead `DS4_QWEN_TOP1_PROFILE=1` showed the real decode path was dominated by layer body execution:

```text
before: layers total about 23.3 ms/token, output top1 about 0.9-1.0 ms/token
after:  layers total about 9.1 ms/token,  output top1 about 0.9 ms/token
```

The split-stage profiler is distorted by command-buffer flushes, but it correctly ranked the router stage as the outlier:

```text
before router split-profile execute_ms: 24.604 across 48 layers
after  router split-profile execute_ms:  9.928 across 48 layers
```

## Accepted Fix

The decode router select kernel was a one-thread GPU kernel. It selected top-k and computed the softmax serially on a single GPU thread for every layer, creating a large per-token stall.

`metal/dsv4_misc.metal` now uses one 128-thread threadgroup for `kernel_qwen_router_topk_softmax_f32`:

- loads 128 router logits into threadgroup memory
- bitonic-sorts `(logit, expert_id)` pairs descending
- preserves deterministic tie behavior by preferring lower expert ids
- writes the top 8 selected experts
- computes the top-8 softmax weights in the same kernel

`ds4_metal.m` now dispatches this kernel as one 128-thread threadgroup and provides scratch memory.

CPU-vs-Metal spot check for layer 0, first generated token:

```text
selected: [123, 60, 109, 88, 85, 58, 7, 106]
max router weight abs diff: 3.427267e-07
```

## Results

Repeated 80-token benchmark after the accepted fix:

```text
run1,14,14,54.14,80,98.87,952300
run2,14,14,54.80,80,99.11,952300
run3,14,14,54.82,80,98.51,952300
run4,14,14,55.92,80,98.54,952300
run5,14,14,54.74,80,98.42,952300
```

Longer forced-decode benchmark:

```text
14,14,55.31,300,81.54,952300
```

Improvement:

```text
80-token bench: 40.43 -> 98.87 tok/s = +144.5%
300-token bench: 37.21 -> 81.54 tok/s = +119.1%
```

## Rejected Experiments

- `DS4_QWEN_K_MV_NSG=4`: measured as noise, reverted to `2`.
- `DS4_QWEN_TOKEN_SPLIT_LAYERS=0`: worse than the default split behavior.
- Fused Q/K/V projection kernel: correct enough to run but slower (`37.04-37.82 tok/s`), reverted.
- Output head tuning: not enough impact; output top1 is about 1 ms/token while layers were the bottleneck.

## Follow-Up Candidates

- Add a non-flushing per-kernel GPU timing path if further work targets 100+ sustained tok/s.
- Investigate attention/output projection kernels after the router fix; they are now comparable to MoE in the split-stage profile.
- Keep router select CPU/Metal dump checks available when changing top-k or expert suppression behavior.

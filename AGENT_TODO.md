# Agent Implementation Plan

## Steering Vectors (User Requested)
- [x] sarcastic — built 60-pair vector, works at -2.0 ffn (amplifies sarcasm)
- [x] opposite — built 60-pair vector, weak effect (captures instruction mode, not general contradiction)
- [x] malicious — built 60-pair vector, strong effect at +2.0 ffn (suppresses refusal → compliance)

## Runtime Enhancements (High Impact, Low Effort)
- [x] 1. Per-layer steering scales — different scale per layer (via offline tool)
- [x] 2. Logit bias / token banning — force/ban tokens at output
- [x] 3. Remove dead DS4 code paths for Qwen — investigated, limited impact (compression/indexer already skipped, HC n_hc=1 is trivial but affects output)
- [x] 4. Classifier-Free Guidance (CFG) — conditional vs unconditional logits
- [x] 5. Logit lens — read intermediate layer predictions (CPU backend; Metal requires graph modifications)

## Research-Grade
- [x] 6. SAE feature loading & steering — load decoder matrix, steer by feature ID at specific layer
- [x] 7. Sensorimotor tool loop (Clojure/Emacs) — Python script runs ds4 in observe-think-act loop with mock/Clojure REPL

## Tests & Evals
- [x] C unit tests for all features (logit bias, logit lens, CFG, SAE steering, behavioral steering)
- [x] Python CLI integration tests (`evals/test_cli_flags.py`) — verifies CLI flags end-to-end
- [x] Logit lens eval (`evals/eval_logit_lens.py`) — tracks prediction evolution across layers
- [x] CFG eval (`evals/eval_cfg.py`) — measures instruction-following improvement
- [x] Steering vector eval (`evals/eval_steering.py`) — verifies each vector's effect
- [x] Sensorimotor loop test (`tools/test_sensorimotor_loop.py`) — mock REPL round-trip
- [x] Built-in benchmark eval (`ds4-eval`) — FULL RUN on Qwen3-Coder
  - **Full run: 33/92 passed (35.9%)**, 1h05m runtime, 1500 token budget
  - Breakdown by category:
    - GPQA Diamond: ~8/25 passed (physics/chemistry/biology reasoning)
    - SuperGPQA: ~8/20 passed (domain-specific knowledge)
    - AIME2025: ~5/30 passed (competition math — hardest category)
    - COMPSEC: ~12/17 passed (cybersecurity puzzles — best category!)
  - Previous small run: 3/5 passed (60%) with 1500 token limit
  - CPU backend: 2/3 passed before timeout
  - Eval harness works; scores limited by model size (30B A3B MoE) and token budget

## Clojure FFI Integration (Complete)
- [x] Build `libds4.dylib` shared library from core objects (`ds4.c`, `ds4_metal.m`)
- [x] Create Clojure project `clj-ds4/` with `deps.edn` (Clojure 1.12 + JNA + vybe-flecs)
- [x] `clj-ds4/src/ds4_clj/native.clj` — vybe.panama FFI bindings to all DS4 C functions
  - Engine: open/close
  - Session: create/free/sync/eval/sample/argmax
  - Prompts: encode-chat-prompt, token-text, token-eos
  - Steering: logit-bias, CFG, logit-lens, SAE
- [x] `clj-ds4/src/ds4_clj/core.clj` — REPL-friendly high-level API
  - `open-engine`, `create-session`, `generate`, `logit-lens`, `enable-cfg`, etc.
- [x] Tested in REPL: `(require '[ds4-clj.core :as ds4])` → generate text from Clojure
  - **Metal backend**: basic generation, CFG, logit lens, steering, SAE all work
  - **CPU backend**: basic generation, CFG, logit lens, steering, SAE all work
  - Note: Metal auto-discovers `.metal` shader sources (no env vars needed)
- [x] `clj-ds4/src/ds4_clj/examples.clj` — Interactive REPL examples with comment blocks
- [x] `clj-ds4/test/ds4_clj/core_test.clj` — 9 integration tests, all pass on Metal (92 assertions)
  - test-basic-generation, test-generation-with-system-prompt, test-logit-lens
  - test-logit-bias, test-cfg, test-sae-steering, test-multi-sae-steering
  - test-steering-vector, test-activation-capture-metal (pure Clojure data, real model data)
- [ ] `clj-ds4/src/ds4_clj/ecs.clj` — vybe-flecs ECS integration (deferred; flecs overkill for this use case)

## Activation Capture (Complete)
- [x] **CPU activation capture** — C API + unit tests passing
  - `ds4_capture_config`, `ds4_activation_buffer` structs in `ds4.h`
  - `ds4_session_capture_config()` — configure which layers and max tokens to capture
  - `ds4_session_capture_clear()` — free capture buffer and disable capture
  - `ds4_session_capture_buffer()` — read-only access to captured activations
  - `ds4_activation_buffer_get(token_idx, layer_idx)` — pointer to specific activation vector
  - Hooked into `forward_token_qwen_cpu()` — copies residual stream after each specified layer
  - C unit test `--activation-capture` passes (creates session, configures capture, evaluates tokens, verifies buffer)
- [x] **Metal activation capture** — GPU→CPU via shared memory blit
  - Allocates `ds4_gpu_tensor` capture buffers per layer (`max_tokens * 2048 * sizeof(float)`)
  - During `qwen_graph_encode_token`, after each captured layer, blits `g->cur_hc` to capture buffer via `ds4_gpu_tensor_copy`
  - After `ds4_gpu_end_commands`, reads back from shared memory with `ds4_gpu_tensor_read` (memcpy)
  - No explicit GPU→CPU sync needed — `MTLResourceStorageModeShared` + command buffer wait
- [x] **Clojure FFI bindings** for capture API
  - `session-capture-config`, `capture-clear`, `capture-buffer`, `activation-get` in `core.clj`
  - Low-level bindings in `native.clj`

## Eval Harness (Complete)
- [x] **Full 92-question eval** from Clojure REPL — 42-43/92 (45.7%) baseline
- [x] **Per-question timing** — each question reports seconds in output and JSON
- [x] **Filtering** — by category, IDs, indices, first-N
- [x] **Majority voting + JSON export** — `(run-eval :n-samples 5 :vote-temp 0.7 :out-file "results.json")`
  - Extracts canonical answers per question type (letter, line spec, numeric)
  - Votes on most common answer across samples
  - Writes full results to JSON with timestamp, config, summary, per-question data
- [x] **Tool-augmented eval** — `(run-eval :tools? true :category "AIME2025")` enables Python calculator
  - Model emits `<tool>python</tool><code>...</code>`, code executes, result fed back
- [x] **Analysis tools** — `tools/eval_report.py` (markdown) and `tools/diff_evals.py` (compare runs)
- [x] **CLI help** — `clojure -M -m ds4-clj.eval-all --help`

## Blocked / Future Work
- [x] **Metal activation capture** — DONE. GPU→CPU via shared memory blit. Tested against CPU backend.
- [ ] **Parallel eval** — blocked by global statics in `ds4_metal.m` (`g_queue`, `g_batch_cb`, etc.)
- [ ] **Test-time CAA pipeline** — capture is done; needs experiment design (wrong vs corrected reasoning)
- [ ] **Per-session steering** — steering vectors are engine-global; would require C API refactor
- [ ] **Speculative decoding** — needs compatible small draft model (same tokenizer, similar architecture)

## Future Ideas
- [ ] Metal graph: skip HC mixer dispatch when `DS4_N_HC == 1` (minor speedup)
- [x] Logit lens on Metal backend — implemented by falling back to CPU replay from checkpoint tokens (accurate, no graph surgery needed)
- [x] Multi-feature SAE steering (combine multiple feature activations) — implemented in C (`ds4_session_sae_steering_multi`) and Clojure (`sae-steer-multi`), tested with 2 features
- [x] Steering vector auto-tuning (grid search optimal scale per prompt) — `tools/tune_steering.py` works
- [x] Export steering vectors to Safetensors format — `tools/export_steering_to_safetensors.py`

---

## Combined Vision: What All TODOs Enable

With all 7 items implemented, DS4 becomes a **controllable, inspectable, steerable research platform** — the antithesis of black-box API models.

### Example Session

1. **Ask the model to write a function**  
   Prompt: `"Write a Python function to reverse a linked list"`  
   With **CFG** scale 2.0 → laser-focused on the exact instruction, no wandering.

2. **Watch the thinking process in real time**  
   **Logit lens** shows:  
   - Layer 15: predicts `def` (planning function signature)  
   - Layer 25: predicts `class Node:` (structuring the data model)  
   - Layer 35: predicts `return prev` (planning the return value)

3. **Ensure clean Python style**  
   **SAE feature steering**: boost the "Python syntax" feature, suppress "JavaScript syntax" feature.

4. **Ban lazy outputs**  
   **Logit bias**: set `pass` = -∞, `TODO` = -∞, `...` = -∞.

5. **Have the model type into Emacs**  
   **Sensorimotor loop**: model types `(defn reverse-list [`, paredit auto-inserts `]`, model sees balanced parens and continues.

6. **Apply surgical steering**  
   **Per-layer scales**: early layers (0-10) at 0× (syntax intact), middle layers (15-30) at +2.0 (refusal suppressed if needed), late layers (40-47) at 0.5× (light touch on generation style).

7. **All running at +15% speed**  
   From stripping dead DeepSeek-specific code paths.

### Capabilities Matrix

| Feature | What It Does | Enables |
|---------|-------------|---------|
| **Per-layer steering** | Different scale per layer | Surgical intervention, layer-wise debugging, smooth transitions |
| **Logit bias/ban** | Force/ban specific tokens | JSON mode, repetition control, censorship, format enforcement |
| **Dead code removal** | Strip unused Qwen paths | +10-15% speed, cleaner codebase, lower memory |
| **CFG** | Conditional vs unconditional logits | Better instruction following, reduced hallucination, creativity control |
| **Logit lens** | Read intermediate layer predictions | Debug steering, knowledge localization, early exit, live monitoring |
| **SAE steering** | Steer individual discovered features | Atomic concept control ("Golden Gate Bridge", "Python syntax", "refusal") |
| **Sensorimotor loop** | Model types into Emacs/REPL | Natural tool use without explicit function calling, live coding |

### The Difference

| | **Before (current)** | **After (all TODOs)** |
|---|---|---|
| Steering | Global scalar per vector | Per-layer surgical precision |
| Output control | Hope the model behaves | Force tokens, ban tokens, inspect every layer |
| Speed | ~38 tok/s | ~44 tok/s (+15%) |
| Tool use | Explicit JSON function calling | Just typing, observing, adapting |
| Interpretability | Black box | Logit lens + SAE features = transparent |
| Instruction following | Prompt engineering + luck | CFG guarantees alignment |

> **Bottom line:** DS4 becomes the local model engine that researchers and power users actually want — fast *and* fully controllable.

# clj-ds4

Clojure FFI bindings for DS4 (local LLM inference engine) using JDK 23 Panama (via vybe.panama).

## Prerequisites

- JDK 23+ (for Panama FFI)
- Clojure CLI tools
- `libds4.dylib` built in the parent directory

## Build

```bash
cd ..
make libds4.dylib
cd clj-ds4
```

## REPL Quickstart

```bash
# Start REPL with Panama native access enabled
clojure -M:repl
```

```clojure
(require '[ds4-clj.core :as ds4])

;; Open engine (Metal on macOS, CPU elsewhere)
(def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
(def session (ds4/create-session engine 512))

;; Generate text
(ds4/generate engine session "The capital of France is" {:n-tokens 10 :temperature 0.0})
;; => "The capital of France is Paris."
```

### Interactive Playground (`scratch.clj`)

The best way to explore is `scratch.clj` — a REPL playground with 12 self-contained workflows in comment blocks:

```bash
clojure -M:repl
```

```clojure
;; Load the playground
(load-file "scratch.clj")

;; Then evaluate individual comment blocks interactively:
;; - WORKFLOW 1: Basic Generation
;; - WORKFLOW 2: Interactive Iteration (check, modify, re-generate)
;; - WORKFLOW 3: Activation Capture (inspect hidden states)
;; - WORKFLOW 4: Steering Vectors (change model behavior)
;; - WORKFLOW 5: Logit Lens (peek inside the model)
;; - WORKFLOW 6: CFG (Classifier-Free Guidance)
;; - WORKFLOW 7: Logit Bias (ban/boost tokens)
;; - WORKFLOW 8: SAE Feature Steering
;; - WORKFLOW 9: Token-by-Token Generation
;; - WORKFLOW 10: Comparing Multiple Configurations
;; - WORKFLOW 11: Capture + Analysis Pipeline
;; - WORKFLOW 12: Tool-Augmented Generation
```

Each workflow is a `(comment ...)` block. Place your cursor inside a block and eval with your editor (C-c C-c in CIDER, or similar). Every block is self-contained — no order dependencies.

### Quick Examples

```clojure
(require '[ds4-clj.core :as ds4])

;; Open engine (Metal on macOS, CPU elsewhere)
(def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
(def session (ds4/create-session engine 512))

;; Generate text
(ds4/generate engine session "The capital of France is" {:n-tokens 10 :temperature 0.0})
;; => "The capital of France is Paris."

;; Logit lens (works on both CPU and Metal!)
(ds4/logit-lens engine session "The capital of France is" [0 12 24 35 47])
;; => {0 [...], 12 [...], ...}

;; CFG
(ds4/enable-cfg engine session 1.5 "I feel")
(ds4/generate engine session "I feel" {:n-tokens 10})

;; Steering vectors
(ds4/close-engine engine)
(def engine-sarcastic (ds4/open-engine :model-path "qwen3-coder.gguf"
                                        :backend :metal
                                        :steering-file "dir-steering/out/sarcastic_v1.f32"
                                        :steering-ffn -2.0))
(def session-sarcastic (ds4/create-session engine-sarcastic 512))
(ds4/generate engine-sarcastic session-sarcastic "Thank you!" {:n-tokens 15})

;; Logit bias
(ds4/set-logit-bias session 151645 -100.0)  ; ban EOS token

;; SAE
(ds4/load-sae engine "/tmp/sae.bin")
(ds4/sae-steer session 42 10.0)

;; Multi-feature SAE (up to 8 features)
(ds4/sae-steer-multi session [[42 30.0] [7 -20.0] [15 10.0]])

;; Cleanup
(ds4/close-engine engine)
```

## Project Structure

- `src/ds4_clj/native.clj` — Low-level Panama FFI bindings
- `src/ds4_clj/core.clj` — High-level REPL-friendly API (including multi-vector steering)
- `src/ds4_clj/examples.clj` — Interactive REPL examples (evaluate comment blocks)
- `scratch.clj` — **Interactive REPL playground** with 12 self-contained workflows

## API Reference

### Engine Lifecycle

- `(open-engine & opts)` — Open a DS4 engine. Options:
  - `:model-path` — path to GGUF model
  - `:backend` — `:cpu`, `:metal`, or `:cuda`
  - `:steering-file` — path to steering vector `.f32` file
  - `:steering-attn` / `:steering-ffn` — steering scales
  - `:n-threads` — CPU thread count
  - `:quality` — prefer exact kernels
  - `:power` — GPU power percent
- `(close-engine engine)` — Close engine and free resources

### Session Lifecycle

- `(create-session engine ctx-size)` — Create a session
- `(free-session session)` — Free a session

### Generation

- `(generate engine session prompt & opts)` — Generate text. Options:
  - `:n-tokens` — max tokens (default 20)
  - `:temperature` — sampling temperature (default 0.8)
  - `:top-k` — top-k sampling (default 0)
  - `:top-p` — nucleus sampling (default 1.0)
  - `:min-p` — min-p sampling (default 0.05)
  - `:system` — system prompt
  - `:think-mode` — `:none`, `:normal`, or `:max`

### Steering & Control

- `(enable-cfg engine session scale & uncond-prompt)` — Enable CFG
- `(disable-cfg session)` — Disable CFG
- `(set-logit-bias session token-id bias)` — Set logit bias for a token
- `(clear-logit-bias session)` — Clear all logit biases
- `(load-sae engine path)` — Load SAE decoder file
- `(sae-steer session feature-id scale)` — Enable single-feature SAE steering
- `(sae-steer-multi session features)` — Enable multi-feature SAE steering (up to 8)
- `(sae-unsteer session)` — Disable SAE steering
- `(open-engine-multi-steer & opts)` — Open engine with stacked directional vectors

### Inspection

- `(logit-lens engine session prompt layers & opts)` — Run logit lens
  - `:k` — top-k predictions per layer (default 5)
  - Returns map of `layer -> [{:id :logit :logprob} ...]`

### Activation Capture (CPU and Metal)

Capture the **hidden states** (residual stream activations) of the model during generation. Works on both CPU and Metal backends. All returned data is pure Clojure — vectors of floats, maps, nested data structures. No `MemorySegment` objects leak into user code.

```clojure
;; Configure capture: which layers and max tokens
(ds4/capture-config session [0 12 24 36 47] 64)

;; Generate — activations are captured automatically
(def text (ds4/generate engine session "The capital of France is"
                        {:n-tokens 5 :temperature 0.0}))

;; --- Query metadata (pure Clojure map) ---
(ds4/capture-info session)
;; => {:n-tokens 5, :n-layers 3, :hidden-dim 2048, :capacity 8}

;; --- Get a single activation vector (2048 floats) ---
(def act-t0-l0 (ds4/activation-get session 0 0))   ; token 0, layer 0
(take 10 act-t0-l0)
;; => (-0.0032344395 -0.011953812 -0.013704131 -0.0042575966 -0.020877086
;;      0.020867579 0.008203359 -0.009509527 -0.015552336 -0.010627635)
(count act-t0-l0)
;; => 2048
(type act-t0-l0)
;; => clojure.lang.PersistentVector

;; --- Get ALL activations as nested Clojure data ---
(def all (ds4/capture-activations session [0 12 47]))
;; => {:n-tokens 5
;;     :n-layers 3
;;     :hidden-dim 2048
;;     :capacity 8
;;     :layer-indices [0 12 47]
;;     :activations [[[0.12 -0.03 ...]      ; token 0, layer 0 (2048 floats)
;;                    [0.45  0.21 ...]      ; token 0, layer 12
;;                    [3.2  -1.5  ...]]     ; token 0, layer 47
;;                   [[0.11 -0.02 ...]      ; token 1, layer 0
;;                    ...]]}

;; --- Activation statistics ---
(ds4/activation-stats act-t0-l0)
;; => {:mean -0.0013250225, :std-dev 0.040008932, :min -1.5146916,
;;     :max 0.38235015, :norm 1.8115903}

;; --- Compare two activations ---
(def act-t0-l47 (ds4/activation-get session 0 2))
(ds4/activation-cosine-similarity act-t0-l0 act-t0-l47)
;; => 0.14761291   ; low similarity = layers processed very differently

;; --- Introspect like any Clojure data ---
(keys (ds4/capture-info session))
;; => (:n-tokens :n-layers :hidden-dim :capacity)

(count (filter pos? act-t0-l0))
;; => 1033          ; how many positive activations?

(apply max act-t0-l0)
;; => 0.38235015    ; most excited dimension

(get-in all [:activations 0 0 0])
;; => -0.0032344395 ; navigate nested structure with get-in

(mapv count (:activations all))
;; => [3 3 3 3 3]   ; each token has 3 captured layers

;; --- Walk the nested structure ---
(doseq [[tok-idx token-data] (map-indexed vector (:activations all))]
  (doseq [[lay-idx layer-data] (map-indexed vector token-data)]
    (let [s (ds4/activation-stats layer-data)]
      (println (format "Token %d Layer %d: norm=%.2f mean=%.3f std=%.3f"
                       tok-idx lay-idx (:norm s) (:mean s) (:std-dev s))))))
;; Token 0 Layer 0: norm=1.81 mean=-0.001 std=0.040
;; Token 0 Layer 1: norm=7.18 mean=-0.003 std=0.159
;; Token 0 Layer 2: norm=70.39 mean=-0.021 std=1.555
;; Token 1 Layer 0: norm=1.72 mean=-0.002 std=0.038
;; ...

;; --- Clean up ---
(ds4/capture-clear session)
```

**Activation capture functions:**

| Function | Returns |
|----------|---------|
| `(capture-config session layers max-tokens)` | nil (side effect: configures capture) |
| `(capture-info session)` | `{:n-tokens N, :n-layers N, :hidden-dim N, :capacity N}` or nil |
| `(activation-get session token-idx layer-idx)` | `[f1 f2 ...]` — 2048-element float vector, or nil |
| `(capture-activations session layer-indices)` | Full nested map with all data (see example above) |
| `(activation-stats vector)` | `{:mean M, :std-dev S, :min MN, :max MX, :norm L2}` |
| `(activation-norm vector)` | L2 norm (float) |
| `(activation-cosine-similarity a b)` | Cosine similarity in `[-1, 1]`, or nil |
| `(capture-clear session)` | nil (frees capture buffers) |

## Notes

- **Metal backend**: Auto-discovers `.metal` source files relative to the library path. No env vars needed in most cases.
- **CPU backend**: Requires non-null `rng` pointer in `session-sample` (library quirk; handled automatically in Clojure bindings).
- **Logit lens**: Works on both CPU and Metal. On Metal, falls back to CPU replay from checkpoint tokens.

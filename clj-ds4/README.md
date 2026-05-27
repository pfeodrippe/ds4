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

## Notes

- **Metal backend**: Auto-discovers `.metal` source files relative to the library path. No env vars needed in most cases.
- **CPU backend**: Requires non-null `rng` pointer in `session-sample` (library quirk; handled automatically in Clojure bindings).
- **Logit lens**: Works on both CPU and Metal. On Metal, falls back to CPU replay from checkpoint tokens.

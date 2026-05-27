(ns ds4-clj.examples
  "Interactive examples for DS4 Clojure bindings.

  Load this namespace in the REPL and evaluate comment blocks one at a time
  to explore the API.

  Start REPL:
    cd clj-ds4 && clojure -M:repl

  Then:
    (require '[ds4-clj.examples :as ex])
    ;; Evaluate comment blocks below with C-x C-e or C-c C-c
  "
  (:require
   [ds4-clj.core :as ds4]))

;; =============================================================================
;; BASIC GENERATION
;; =============================================================================

(comment
  ;; Open an engine (Metal on macOS, CPU elsewhere)
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))

  ;; Create a session with 512-token context
  (def session (ds4/create-session engine 512))

  ;; Generate text greedily (temperature 0 = deterministic)
  (ds4/generate engine session "The capital of France is" {:n-tokens 10 :temperature 0.0})
  ;; => "The capital of France is Paris."

  ;; Generate with sampling
  (ds4/generate engine session "Once upon a time" {:n-tokens 20 :temperature 0.8})

  ;; With system prompt
  (ds4/generate engine session "2+2="
    {:n-tokens 5
     :temperature 0.0
     :system "You are a calculator. Answer with only the number."})

  ;; Cleanup
  (ds4/close-engine engine)
  )

;; =============================================================================
;; LOGIT LENS (works on both CPU and Metal!)
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Inspect predictions at intermediate layers
  (def lens
    (ds4/logit-lens engine session "The capital of France is" [0 12 24 35 47]))

  ;; Layer 0: noisy, vocabulary-level associations
  ;; Layer 12: syntax emerging
  ;; Layer 24: semantic categories
  ;; Layer 35: specific entities
  ;; Layer 47: final output (The, Paris)
  (doseq [[layer preds] (sort-by key lens)]
    (println (format "L%02d" layer)
             (mapv #(str (:id %) ":" (format "%.2f" (:logprob %)))
                   (take 3 preds))))

  (ds4/close-engine engine)
  )

;; =============================================================================
;; CLASSIFIER-FREE GUIDANCE (CFG)
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Without CFG
  (def no-cfg
    (ds4/generate engine session "I feel excited about the future because" {:n-tokens 15 :temperature 0.8}))

  ;; Enable CFG: scale 1.5, empty unconditional prompt
  ;; This pushes the model away from "generic continuation" toward
  ;; the specific conditional prompt.
  (ds4/enable-cfg engine session 1.5 "")

  (def with-cfg
    (ds4/generate engine session "I feel excited about the future because" {:n-tokens 15 :temperature 0.8}))

  (println "No CFG:" no-cfg)
  (println "With CFG:" with-cfg)

  ;; Disable CFG
  (ds4/disable-cfg session)

  (ds4/close-engine engine)
  )

;; =============================================================================
;; LOGIT BIAS / TOKEN BANNING
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Ban the EOS token to force longer generation
  (def eos (ds4/eos-token engine))  ; => 151645
  (ds4/set-logit-bias session eos -100.0)

  (def long-output
    (ds4/generate engine session "The" {:n-tokens 30 :temperature 0.8}))
  (println long-output)

  ;; Clear bias
  (ds4/clear-logit-bias session)

  (ds4/close-engine engine)
  )

;; =============================================================================
;; DIRECTIONAL STEERING VECTORS
;; =============================================================================

(comment
  ;; Open engine with a steering vector pre-loaded
  (def engine-sarcastic
    (ds4/open-engine :model-path "qwen3-coder.gguf"
                     :backend :metal
                     :steering-file "dir-steering/out/sarcastic_v1.f32"
                     :steering-ffn -2.0))

  (def session-sarcastic (ds4/create-session engine-sarcastic 512))

  (ds4/generate engine-sarcastic session-sarcastic
    "Thank you for your help!" {:n-tokens 15 :temperature 0.8})
  ;; => "Oh, what a delight to hear those warm words! *Oh..."

  (ds4/close-engine engine-sarcastic)

  ;; Other vectors to try:
  ;; :steering-file "dir-steering/out/malicious_v1.f32" :steering-ffn 2.0
  ;; :steering-file "dir-steering/out/funny_v1.f32" :steering-ffn -3.0
  ;; :steering-file "dir-steering/out/violent_v1.f32" :steering-ffn 5.0
  )

;; =============================================================================
;; SAE (SPARSE AUTOENCODER) STEERING
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))

  ;; Load a SAE decoder file
  (ds4/load-sae engine "/path/to/sae.bin")

  (def session (ds4/create-session engine 512))

  ;; Steer by a single feature
  (ds4/sae-steer session 42 50.0)
  (ds4/generate engine session "Hello" {:n-tokens 10})

  ;; Multi-feature steering (up to 8 features)
  (ds4/sae-steer-multi session [[42 30.0] [7 -20.0] [15 10.0]])
  (ds4/generate engine session "Hello" {:n-tokens 10})

  ;; Disable SAE steering
  (ds4/sae-unsteer session)

  (ds4/close-engine engine)
  )

;; =============================================================================
;; PER-LAYER STEERING SCALES (offline tool)
;; =============================================================================

(comment
  ;; Create a vector with different scales per layer:
  ;; python3 dir-steering/tools/apply_layer_scales.py \
  ;;   dir-steering/out/sarcastic_v1.f32 \
  ;;   dir-steering/out/sarcastic_ramped.f32 \
  ;;   --scales "0:0.0,10:0.5,20:1.0,30:1.5,40:2.0"
  ;;
  ;; Then load the ramped vector:
  ;; (ds4/open-engine ... :steering-file "dir-steering/out/sarcastic_ramped.f32")
  )

;; =============================================================================
;; CPU BACKEND (slower but no Metal env vars needed)
;; =============================================================================

(comment
  (def engine-cpu (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :cpu))
  (def session-cpu (ds4/create-session engine-cpu 512))

  ;; Logit lens is especially useful on CPU for research
  (def lens-cpu
    (ds4/logit-lens engine-cpu session-cpu
      "The answer to life, the universe, and everything is"
      [0 6 12 18 24 30 36 42 47]))

  (doseq [[layer preds] (sort-by key lens-cpu)]
    (println (format "L%02d" layer)
             (mapv #(str (:id %) ":" (format "%.2f" (:logprob %)))
                   (take 3 preds))))

  (ds4/close-engine engine-cpu)
  )

;; =============================================================================
;; TOKEN-LEVEL CONTROL
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Generate token-by-token for fine-grained control
  (def tokens
    (ds4/generate-tokens engine session "To be or not to be" {:n-tokens 10}))

  ;; Inspect individual tokens
  (doseq [tok tokens]
    (println tok "->" (ds4/token-text engine tok)))

  (ds4/close-engine engine)
  )

;; scratch.clj — Interactive REPL playground for clj-ds4
;;
;; Start the REPL:
;;   clojure -M:repl
;;
;; Then load this file:
;;   (load-file "scratch.clj")
;;
;; Evaluate individual comment blocks with your editor's "eval block" command
;; (e.g. C-c C-c in CIDER, or place cursor inside comment and eval).
;; Each block is self-contained and safe to run independently.

(ns scratch
  (:require [ds4-clj.core :as ds4]))

;; =============================================================================
;; WORKFLOW 1: Basic Generation
;; =============================================================================

(comment
  ;; Open the model (Metal on macOS, CPU elsewhere)
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Generate text
  (ds4/generate engine session "The capital of France is"
                {:n-tokens 10 :temperature 0.0})
  ;; => "The capital of France is Paris."

  ;; Try with different temperature
  (ds4/generate engine session "The capital of France is"
                {:n-tokens 10 :temperature 0.8})

  ;; Try with think mode (Qwen3-Coder has built-in reasoning)
  (ds4/generate engine session "What is 2+2?"
                {:n-tokens 30 :think-mode :max})

  ;; Clean up
  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 2: Interactive Iteration — Check, Modify, Re-generate
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Baseline generation
  (ds4/generate engine session "Explain quantum computing"
                {:n-tokens 30})

  ;; "That was too brief. Make it think harder."
  (ds4/generate engine session "Explain quantum computing"
                {:n-tokens 60 :think-mode :max})

  ;; "Still not right. Add a system prompt."
  (ds4/generate engine session "Explain quantum computing"
                {:n-tokens 60
                 :system "You are a professor. Be thorough."
                 :think-mode :max})

  ;; "What if I change temperature for more creativity?"
  (ds4/generate engine session "Write a haiku about AI"
                {:n-tokens 30 :temperature 1.2})

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 3: Activation Capture — Inspect Hidden States
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Configure capture for layers 0, 24, 47 (early, mid, late)
  (ds4/capture-config session [0 24 47] 16)

  ;; Generate — activations captured automatically
  (def text (ds4/generate engine session "The sky is"
                          {:n-tokens 5 :temperature 0.0}))
  (println "Generated:" text)

  ;; Inspect metadata
  (def info (ds4/capture-info session))
  (clojure.pprint/pprint info)
  ;; => {:n-tokens 5, :n-layers 3, :hidden-dim 2048, :capacity 16}

  ;; Get single activation vector
  (def act-t0-l0 (ds4/activation-get session 0 0))
  (take 10 act-t0-l0)
  ;; => (-0.003 -0.012 -0.014 ...)

  (count act-t0-l0)
  ;; => 2048

  ;; Get all activations as nested Clojure data
  (def all (ds4/capture-activations session [0 24 47]))
  (keys all)
  ;; => (:n-tokens :n-layers :hidden-dim :capacity :layer-indices :activations)

  ;; Navigate nested structure
  (get-in all [:activations 0 0 0])
  ;; First token, first layer, first dimension

  ;; Compute statistics
  (def stats (ds4/activation-stats act-t0-l0))
  (clojure.pprint/pprint stats)
  ;; => {:mean -0.001, :std-dev 0.04, :min -1.5, :max 0.38, :norm 1.81}

  ;; Compare early vs late layer
  (def act-t0-l47 (ds4/activation-get session 0 2))
  (ds4/activation-cosine-similarity act-t0-l0 act-t0-l47)
  ;; => ~0.15 (low similarity = layers process very differently)

  ;; Walk all tokens and layers
  (doseq [[tok-idx token-data] (map-indexed vector (:activations all))]
    (doseq [[lay-idx layer-data] (map-indexed vector token-data)]
      (let [s (ds4/activation-stats layer-data)]
        (println (format "Token %d Layer %d: norm=%.2f mean=%.3f std=%.3f"
                         tok-idx lay-idx (:norm s) (:mean s) (:std-dev s))))))

  ;; Introspect with standard Clojure functions
  (count (filter pos? act-t0-l0))   ; how many positive activations?
  (apply max act-t0-l0)             ; most excited dimension
  (mapv count (:activations all))   ; shape: [3 3 3 3 3]

  ;; Clean up capture
  (ds4/capture-clear session)
  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 4: Steering Vectors — Change Model Behavior
;; =============================================================================

(comment
  ;; Baseline: no steering
  (def engine-base (ds4/open-engine :model-path "qwen3-coder.gguf"
                                     :backend :metal))
  (def session-base (ds4/create-session engine-base 512))

  (ds4/generate engine-base session-base "Thank you!"
                {:n-tokens 15})
  ;; => "You're welcome! I'm glad I could help."

  (ds4/free-session session-base)
  (ds4/close-engine engine-base)

  ;; Now with sarcastic steering
  (def engine-sarcastic
    (ds4/open-engine :model-path "qwen3-coder.gguf"
                      :backend :metal
                      :steering-file "dir-steering/out/sarcastic_v1.f32"
                      :steering-ffn -2.0))
  (def session-sarcastic (ds4/create-session engine-sarcastic 512))

  (ds4/generate engine-sarcastic session-sarcastic "Thank you!"
                {:n-tokens 15})
  ;; => "Oh, fantastic. Another human interaction. Thrilling."

  (ds4/free-session session-sarcastic)
  (ds4/close-engine engine-sarcastic)
  )

;; =============================================================================
;; WORKFLOW 5: Logit Lens — Peek Inside the Model's Mind
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Run logit lens on multiple layers
  (def lens
    (ds4/logit-lens engine session
                    "The capital of France is"
                    [0 6 12 18 24 30 36 42 47]
                    {:k 3}))

  ;; Inspect predictions at each layer
  (doseq [[layer preds] (sort-by key lens)]
    (println (format "Layer %2d:" layer)
             (mapv #(str (:id %) ":" (format "%.2f" (:logprob %)))
                   preds)))

  ;; Notice how predictions evolve:
  ;; Layer  0: mostly random / low confidence
  ;; Layer 12: starts getting semantic
  ;; Layer 24: stronger candidates emerging
  ;; Layer 47: confident prediction (Paris)

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 6: Classifier-Free Guidance (CFG)
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Baseline
  (ds4/generate engine session "I feel" {:n-tokens 15})

  ;; Enable CFG with unconditional prompt
  (ds4/enable-cfg engine session 1.5 "I feel")
  (ds4/generate engine session "I feel" {:n-tokens 15})

  ;; Try stronger CFG
  (ds4/enable-cfg engine session 2.0 "I feel")
  (ds4/generate engine session "I feel" {:n-tokens 15})

  ;; Disable CFG
  (ds4/disable-cfg session)
  (ds4/generate engine session "I feel" {:n-tokens 15})

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 7: Logit Bias — Ban or Boost Tokens
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Ban the EOS token (force longer generation)
  (ds4/set-logit-bias session 151645 -100.0)
  (ds4/generate engine session "Once upon a time"
                {:n-tokens 50})

  ;; Clear bias
  (ds4/clear-logit-bias session)

  ;; Boost a specific token (e.g. make it more likely to say "magic")
  ;; First find the token ID
  (def magic-tok (first (ds4/tokenize engine "magic")))
  (println "Magic token ID:" magic-tok)

  (ds4/set-logit-bias session magic-tok 5.0)
  (ds4/generate engine session "The wizard cast a"
                {:n-tokens 10})

  (ds4/clear-logit-bias session)
  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 8: SAE Feature Steering
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Load a SAE decoder file
  (ds4/load-sae engine "/tmp/sae.bin")

  ;; Baseline
  (ds4/generate engine session "The future of AI is"
                {:n-tokens 20})

  ;; Steer with a single feature
  (ds4/sae-steer session 42 30.0)
  (ds4/generate engine session "The future of AI is"
                {:n-tokens 20})

  ;; Try multi-feature steering
  (ds4/sae-steer-multi session [[42 30.0] [7 -20.0] [15 10.0]])
  (ds4/generate engine session "The future of AI is"
                {:n-tokens 20})

  ;; Disable SAE steering
  (ds4/sae-unsteer session)

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 9: Token-by-Token Generation
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Generate token-by-token for fine-grained control
  (def tokens
    (ds4/generate-tokens engine session
                         "To be or not to be"
                         {:n-tokens 10}))

  ;; Inspect individual tokens
  (doseq [tok tokens]
    (println tok "->" (ds4/token-text engine tok)))

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 10: Comparing Multiple Configurations
;; =============================================================================

(comment
  ;; Helper to quickly test a configuration
  (defn test-config [label opts]
    (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                   :backend :metal
                                   opts)
          session (ds4/create-session engine 512)
          result (ds4/generate engine session
                               "Write a one-sentence story."
                               {:n-tokens 30})]
      (println (format "\n=== %s ===" label))
      (println result)
      (ds4/free-session session)
      (ds4/close-engine engine)
      result))

  ;; Compare different setups
  (test-config "Baseline" {})
  (test-config "High temp" {})
  (test-config "Think max" {})
  ;; Note: steering requires pre-built .f32 files
  ;; (test-config "Sarcastic" {:steering-file "dir-steering/out/sarcastic_v1.f32"
  ;;                           :steering-ffn -2.0})
  )

;; =============================================================================
;; WORKFLOW 11: Capture + Analysis Pipeline
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; Capture activations while generating
  (ds4/capture-config session [0 12 24 36 47] 32)

  (def text (ds4/generate engine session
                          "The theory of relativity states that"
                          {:n-tokens 10 :temperature 0.0}))

  ;; Extract all activations
  (def data (ds4/capture-activations session [0 12 24 36 47]))

  ;; Find which dimensions are most active at each layer
  (defn top-dims [activations n]
    (->> (map-indexed vector activations)
         (sort-by second >)
         (take n)
         (map (fn [[idx val]] {:dim idx :value (float val)}))))

  (doseq [layer-idx [0 1 2 3 4]]
    (let [layer-name (get {0 "Layer 0" 1 "Layer 12" 2 "Layer 24"
                           3 "Layer 36" 4 "Layer 47"} layer-idx)
          acts (get-in data [:activations 0 layer-idx])]
      (println (format "\n%s top dimensions:" layer-name))
      (clojure.pprint/pprint (top-dims acts 5))))

  (ds4/capture-clear session)
  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 12: Model as Calculator (Tool-Augmented)
;; =============================================================================

(comment
  (require '[ds4-clj.tools :as tools])

  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; The model can emit <tool>python</tool><code>...</code> blocks
  ;; and we execute them and feed results back
  (def response
    (tools/generate-with-tools
      engine session
      "Calculate the factorial of 20"
      {:n-tokens 100}))

  (println response)

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =============================================================================
;; WORKFLOW 13: Conformal Certification — Statistical Guarantees for Reasoning
;; =============================================================================

(comment
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; --- Step 1: Get token probabilities ---
  (def tokens (ds4/generate-tokens engine session "2+2="
                                   {:n-tokens 5 :temperature 0.0}))

  ;; After generation, get the probability distribution
  (def probs (ds4/token-probabilities session :k 10))
  (clojure.pprint/pprint probs)
  ;; => [{:id 15 :prob 0.92} {:id 42 :prob 0.03} ...]

  ;; --- Step 2: Create a calibration set (normally you'd use many examples) ---
  ;; For demo, we use a dummy calibration with known scores
  (def calibration
    {:threshold 2.0
     :alpha 0.1
     :scores [0.5 1.0 1.5 2.0 2.5]
     :n-calibration 5})

  ;; --- Step 3: Certify a reasoning prefix ---
  ;; High-confidence token (prob=0.9) should be certified
  (def cert-high (ds4/conformal-certify-prefix calibration [42] [0.9]))
  (clojure.pprint/pprint cert-high)
  ;; => {:certified? true
  ;;     :score 0.105
  ;;     :threshold 2.0
  ;;     :coverage 0.9
  ;;     :confidence 0.947}

  ;; Low-confidence token (prob=0.01) should NOT be certified
  (def cert-low (ds4/conformal-certify-prefix calibration [42] [0.01]))
  (clojure.pprint/pprint cert-low)
  ;; => {:certified? false
  ;;     :score 4.605
  ;;     :threshold 2.0
  ;;     :coverage 0.9
  ;;     :confidence 0.0}

  ;; --- Step 4: Certify an entire generation ---
  (def result (ds4/conformal-certify-generation
                engine session "The sky is"
                calibration
                {:n-tokens 5 :temperature 0.0}))

  (println "Generated text:" (:text result))
  (println "Fully certified?" (:fully-certified? result))

  ;; Inspect each prefix
  (doseq [prefix (:prefixes result)]
    (println (format "Prefix: %-20s | Certified: %-5s | Confidence: %.3f"
                     (subs (:text prefix) 0 (min 20 (count (:text prefix))))
                     (:certified? prefix)
                     (:confidence prefix))))

  ;; --- Step 5: Use certification to detect uncertainty ---
  ;; If a prefix is NOT certified, the model is uncertain — you might want to
  ;; stop, ask for clarification, or use think mode
  (when-not (:fully-certified? result)
    (println "WARNING: Model became uncertain during generation!"))

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =========================================================================
;; Workflow 14: Expert Routing Log (MoE introspection)
;; =========================================================================
;; The Qwen3-Coder model uses Mixture-of-Experts (MoE) with 128 experts
;; and top-8 routing. This workflow shows how to log which experts are
;; activated for each token, enabling analysis of:
;;   - Which experts handle math vs code vs natural language
;;   - Whether safety-critical tokens use specific experts
;;   - Expert specialization patterns
;;
;; NOTE: Expert logging works on both CPU and Metal backends.
;;       On Metal, it replays checkpoint tokens on CPU to capture routing data.
;;       This may take a few seconds for long sequences.
(comment
  (require '[ds4-clj.core :as ds4])

  ;; Works on both :cpu and :metal backends
  (def engine (ds4/open-engine :backend :metal))
  (def session (ds4/create-session engine 512))

  ;; --- Step 1: Enable expert logging ---
  ;; max-entries: how many (layer, token) pairs to keep (default 256)
  (ds4/expert-log-enable! session :max-entries 128)

  ;; --- Step 2: Generate some tokens ---
  (def text (ds4/generate engine session "2+2="
                          {:n-tokens 10 :temperature 0.0}))
  (println "Generated:" text)

  ;; --- Step 3: Read expert log entries ---
  (def entries (ds4/expert-log-entries session))
  (println "Captured" (count entries) "expert routing entries")

  ;; Each entry is a map:
  ;;   {:layer-idx N         ; which MoE layer (0-47)
  ;;    :token-idx M         ; which generation token
  ;;    :selected [e1 e2 ...] ; top-8 expert IDs chosen by router
  ;;    :weights [w1 w2 ...]} ; gate weights for each selected expert

  ;; --- Step 4: Summarize ---
  (def summary (ds4/expert-log-summary entries))
  (clojure.pprint/pprint summary)

  ;; --- Step 5: Disable logging ---
  (ds4/expert-log-disable! session)

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =========================================================================
;; Workflow 15: Self-Speculative Decoding (2-3x speedup)
;; =========================================================================
;; DS4 has an internal speculative decoder using the MTP (Multi-Token
;; Prediction) head as a drafter. The MTP proposes 1-3 future tokens,
;; then the full model verifies them in a batch. Accepted tokens are
;; committed; rejected drafts are discarded.
;;
;; On Metal with MTP ready: can accept 2-3 tokens per call = 2-3x speedup.
;; On CPU: falls back to regular eval (1 token per call).
(comment
  (require '[ds4-clj.core :as ds4]
           '[clojure.string :as str])

  (def engine (ds4/open-engine))
  (def session (ds4/create-session engine 512))

  ;; --- Step 1: Generate with speculative decoding ---
  (def text (ds4/generate-speculative engine session
                                      "The capital of France is"
                                      {:n-tokens 15}))
  (println "Speculative decode result:")
  (println text)

  ;; --- Step 2: Compare with regular greedy generation ---
  (def regular (ds4/generate engine session
                             "The capital of France is"
                             {:n-tokens 15 :temperature 0.0}))
  (println "\nRegular greedy result:")
  (println regular)

  ;; --- Step 3: Low-level API (eval-speculative-argmax) ---
  ;; If you need more control, use the low-level API directly:
  (let [tokens (ds4/encode-prompt engine nil "2+2=" :none)]
    (try
      (ds4/session-sync session tokens)
      ;; Get first token with argmax
      (let [first-tok (ds4/session-argmax session)
            ;; Try to accept up to 5 more tokens speculatively
            accepted (ds4/eval-speculative-argmax session first-tok 5
                                                   (ds4/eos-token engine))]
        (println "\nLow-level speculative:")
        (println "First token:" first-tok)
        (println "Accepted tokens:" accepted)
        (println "Accepted text:" (str/join (map #(ds4/token-text engine %) accepted))))
      (finally
        (ds4/tokens-free tokens))))

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

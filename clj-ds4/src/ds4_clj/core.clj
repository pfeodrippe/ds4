(ns ds4-clj.core
  "High-level REPL-friendly API for DS4."
  (:require
   [clojure.string :as str]
   [ds4-clj.native :as n]
   [vybe.panama :as vp]))

;; --- Engine lifecycle ---

(defn- resolve-model-path
  "Resolve a model path relative to DS4_DIR if it's not absolute."
  [path]
  (if (.isAbsolute (clojure.java.io/file path))
    path
    (str (or (System/getenv "DS4_DIR")
             (str (System/getProperty "user.home") "/dev/ds4"))
         "/" path)))

(defn open-engine
  "Open a DS4 engine.

  Options:
    :model-path        - path to GGUF model (default: qwen3-coder.gguf)
    :backend           - :cpu, :metal, or :cuda (default: :metal on macOS, :cpu elsewhere)
    :steering-file     - path to steering vector file
    :steering-attn     - attention steering scale (default: 0)
    :steering-ffn      - FFN steering scale (default: 0)
    :n-threads         - CPU threads (default: 0 = auto)
    :quality           - prefer exact kernels (default: false)
    :power             - GPU power percent 1..100 (default: 100)
    :mtp-path          - optional MTP model path
    :mtp-draft-tokens  - MTP draft tokens (default: 1)
    :mtp-margin        - MTP margin (default: 3)"
  [& {:keys [model-path backend steering-file steering-attn steering-ffn
             n-threads quality power mtp-path mtp-draft-tokens mtp-margin]
      :or {model-path "qwen3-coder.gguf"
           backend (if (str/includes? (str/lower-case (System/getProperty "os.name")) "mac")
                :metal
                :cpu)
           steering-attn 0.0
           steering-ffn 0.0
           n-threads 0
           quality false
           power 100
           mtp-draft-tokens 1
           mtp-margin 3.0}}]
  (let [opts (n/DS4EngineOptions
              {:model_path (resolve-model-path model-path)
               :mtp_path (or mtp-path "")
               :directional_steering_file (or (when steering-file (resolve-model-path steering-file)) "")
               :directional_steering_attn (float steering-attn)
               :directional_steering_ffn (float steering-ffn)
               :backend (int (get n/backend backend 0))
               :n_threads (int n-threads)
               :mtp_draft_tokens (int mtp-draft-tokens)
               :mtp_margin (float mtp-margin)
               :n_steering_vectors (int 0)
               :power_percent (int power)
               :warm_weights (byte 0)
               :quality (byte (if quality 1 0))})]
    (n/engine-open (vp/mem opts))))

(defn open-engine-multi-steer
  "Open a DS4 engine with multiple steering vectors stacked (up to 8).

  vectors: sequence of maps, each with:
    :file     - path to .f32 steering vector
    :attn     - attention scale (default 0)
    :ffn      - FFN scale (default 0)

  Example:
    (open-engine-multi-steer :model-path \"qwen3-coder.gguf\"
                             :backend :metal
                             :vectors [{:file \"refusal.f32\" :ffn 2.0}
                                       {:file \"sarcastic.f32\" :ffn -1.5}])
  "
  [& {:keys [model-path backend vectors
             n-threads quality power mtp-path mtp-draft-tokens mtp-margin]
      :or {model-path "qwen3-coder.gguf"
           backend (if (str/includes? (str/lower-case (System/getProperty "os.name")) "mac")
                     :metal
                     :cpu)
           n-threads 0
           quality false
           power 100
           mtp-draft-tokens 1
           mtp-margin 3.0}}]
  (let [n-vec (min (count vectors) 8)
        opts (n/DS4EngineOptions
              {:model_path (resolve-model-path model-path)
               :mtp_path (or mtp-path "")
               :directional_steering_file ""
               :directional_steering_attn 0.0
               :directional_steering_ffn 0.0
               :backend (int (get n/backend backend 0))
               :n_threads (int n-threads)
               :mtp_draft_tokens (int mtp-draft-tokens)
               :mtp_margin (float mtp-margin)
               :n_steering_vectors (int n-vec)
               :power_percent (int power)
               :warm_weights (byte 0)
               :quality (byte (if quality 1 0))})
        ^java.lang.foreign.MemorySegment mem (vp/mem opts)]
    ;; steering_vectors array starts at offset 48, each element is 24 bytes
    (doseq [[i v] (map-indexed vector vectors)]
      (when (< i 8)
        (let [base (+ 48 (* i 24))
              file-path (when-let [f (:file v)] (resolve-model-path f))]
          ;; file pointer at offset 0
          (when file-path
            (.set mem java.lang.foreign.ValueLayout/ADDRESS base (vp/try-string file-path)))
          ;; attn_scale at offset 8
          (.set mem java.lang.foreign.ValueLayout/JAVA_FLOAT (+ base 8) (float (get v :attn 0.0)))
          ;; ffn_scale at offset 12
          (.set mem java.lang.foreign.ValueLayout/JAVA_FLOAT (+ base 12) (float (get v :ffn 0.0)))
          ;; layer_scales_file at offset 16 (nil for now)
          )))
    (n/engine-open mem)))

(defn close-engine
  "Close a DS4 engine."
  [engine]
  (n/engine-close engine))

;; --- Session lifecycle ---

(defn create-session
  "Create a session with the given context size."
  [engine ctx-size]
  (n/session-create engine ctx-size))

(defn free-session
  "Free a session."
  [session]
  (n/session-free session))

;; --- Prompt handling ---

(defn encode-prompt
  "Encode a chat prompt. Returns a tokens segment."
  ([engine user]
   (encode-prompt engine nil user :none))
  ([engine system user think-mode]
   (let [think-int (get n/think-mode think-mode 0)]
     (n/encode-chat-prompt engine
                           (if system (vp/try-string system) vp/null)
                           (vp/try-string user)
                           think-int))))

;; --- Generation ---

(defn generate-tokens
  "Generate tokens from a prompt. Returns a vector of token ids.

  Options:
    :n-tokens     - max tokens to generate (default: 20)
    :temperature  - sampling temperature (default: 0.8)
    :top-k        - top-k sampling (default: 0 = disabled)
    :top-p        - nucleus sampling (default: 1.0)
    :min-p        - min-p sampling (default: 0.05)
    :system       - system prompt (default: nil)
    :think-mode   - :none, :normal, or :max (default: :none)"
  [engine session prompt & {:keys [n-tokens temperature top-k top-p min-p system think-mode]
                            :or {n-tokens 20
                                 temperature 0.8
                                 top-k 0
                                 top-p 1.0
                                 min-p 0.05
                                 think-mode :none}}]
  (let [tokens (encode-prompt engine system prompt think-mode)]
    (try
      (n/session-sync session tokens)
      (let [eos (n/token-eos engine)
            ;; Use argmax for temp=0 (deterministic, much faster - no RNG alloc per token)
            sample-fn (if (== temperature 0.0)
                        n/session-argmax
                        #(n/session-sample % temperature top-k top-p min-p))
            result (loop [generated []]
                     (if (>= (count generated) n-tokens)
                       generated
                       (let [token (sample-fn session)]
                         (if (= token eos)
                           generated
                           (do
                             (n/session-eval session token)
                             (recur (conj generated token)))))))
            _ (n/tokens-free tokens)]
        result)
      (catch Exception e
        (n/tokens-free tokens)
        (throw e)))))

(defn generate
  "Generate text from a prompt. Returns a string.

  Same options as generate-tokens."
  [engine session prompt & opts]
  (let [tokens (apply generate-tokens engine session prompt opts)]
    (str/join (map #(n/token-text engine %) tokens))))

;; --- Steering ---

(defn add-steering
  "Add a steering vector to an engine. The engine must be reopened.
  This modifies the engine options; create a new session after calling."
  [engine vector-file ffn-scale]
  (throw (ex-info "Steering vectors must be set at engine open time. Use :steering-file and :steering-ffn options in open-engine."
                  {})))

;; --- Logit bias ---

(defn set-logit-bias
  "Set logit bias for a specific token."
  [session token-id bias]
  (n/set-logit-bias session token-id bias))

(defn clear-logit-bias
  "Clear all logit biases."
  [session]
  (n/clear-logit-bias session))

;; --- CFG ---

(defn enable-cfg
  "Enable Classifier-Free Guidance.

  engine: the engine that owns the session.
  scale: 0 disables CFG. Typical: 1.0-2.0.
  uncond-prompt: the unconditional prompt (default: empty string)."
  ([engine session scale]
   (enable-cfg engine session scale ""))
  ([engine session scale uncond-prompt]
   (let [uncond-tokens (encode-prompt engine uncond-prompt)]
     (try
       (n/set-cfg session scale uncond-tokens)
       (finally
         (n/tokens-free uncond-tokens))))))

(defn disable-cfg
  "Disable CFG."
  [session]
  (n/clear-cfg session))

;; --- Logit lens ---

(defn logit-lens
  "Run logit lens on a prompt. Returns a map of layer -> top-k predictions.

  layers: sequence of layer indices (e.g. [0 12 24 35 47])
  k: number of top predictions per layer (default: 5)"
  [engine session prompt layers & {:keys [k system think-mode]
                                   :or {k 5
                                        think-mode :none}}]
  (let [tokens (encode-prompt engine system prompt think-mode)]
    (try
      (n/session-sync session tokens)
      (let [results (into {}
                          (map (fn [layer]
                                 [layer (n/layer-logprobs session layer k)]))
                          layers)]
        (n/tokens-free tokens)
        results)
      (catch Exception e
        (n/tokens-free tokens)
        (throw e)))))

;; --- SAE ---

(defn load-sae
  "Load a SAE decoder file into the engine."
  [engine path]
  (n/load-sae engine path))

(defn sae-steer
  "Enable SAE steering on a session."
  [session feature-id scale]
  (n/sae-steering-set session feature-id scale))

(defn sae-steer-multi
  "Enable multi-feature SAE steering on a session.

  features: sequence of [feature-id scale] pairs, e.g. [[5 10.0] [3 -5.0]]
  Max 8 features."
  [session features]
  (n/sae-steering-multi session features))

(defn sae-unsteer
  "Disable SAE steering."
  [session]
  (n/sae-steering-clear session))

;; --- Model (immutable configuration) ---

(defrecord Model [engine config])
;; config keys: :steering {:file ... :ffn ...} :think :none|:normal|:max :system nil|str

(defn make-model
  "Create a base Model from an open engine."
  [engine]
  (->Model engine {:steering nil :think :none :system nil}))

(defn with-steering
  "Return a new Model with steering applied.
  NOTE: steering requires reopening the engine. Use reopen-model to apply."
  [model file ffn-scale]
  (assoc-in model [:config :steering] {:file file :ffn ffn-scale}))

(defn with-think
  "Return a new Model with think mode."
  [model mode]
  (assoc-in model [:config :think] mode))

(defn with-system
  "Return a new Model with system prompt."
  [model system-prompt]
  (assoc-in model [:config :system] system-prompt))

(defn reopen-model
  "Reopen a Model's engine to apply configuration changes (e.g. steering).
  Returns a new Model with a fresh engine. Closes the old engine."
  [model]
  (let [cfg (:config model)
        steering (:steering cfg)
        new-engine (if steering
                     (open-engine :steering-file (:file steering)
                                  :steering-ffn (:ffn steering))
                     (open-engine))]
    ;; Close old engine
    (when-let [old (:engine model)]
      (close-engine old))
    (assoc model :engine new-engine)))

(defn generate-model
  "Generate text from a Model. Creates and manages its own session.

  Options:
    :n-tokens     - max tokens (default: 20)
    :temperature  - sampling temp (default: 0.8)"
  [model prompt & {:keys [n-tokens temperature]
                    :or {n-tokens 20 temperature 0.8}}]
  (let [engine (:engine model)
        cfg (:config model)
        session (create-session engine 4096)]
    (try
      (generate engine session prompt
                :n-tokens n-tokens :temperature temperature
                :system (:system cfg)
                :think-mode (:think cfg))
      (finally
        (free-session session)))))

(defn generate-with-session
  "Generate text from a Model using an existing session.
  Much faster than generate-model for repeated calls.
  The session is NOT freed — caller must manage it.

  Options:
    :n-tokens     - max tokens (default: 20)
    :temperature  - sampling temp (default: 0.8)"
  [model session prompt & {:keys [n-tokens temperature]
                            :or {n-tokens 20 temperature 0.8}}]
  (let [engine (:engine model)
        cfg (:config model)]
    (generate engine session prompt
              :n-tokens n-tokens :temperature temperature
              :system (:system cfg)
              :think-mode (:think cfg))))

;; --- Utils ---

(defn eos-token
  "Get the EOS token id."
  [engine]
  (n/token-eos engine))

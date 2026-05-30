(ns ds4-clj.core
  "High-level REPL-friendly API for DS4."
  (:require
   [clojure.string :as str]
   [ds4-clj.native :as n]
   [vybe.panama :as vp])
  (:import
   [java.lang.foreign MemorySegment ValueLayout]))

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

;; --- Activation Capture ---
;; All activation capture functions return pure Clojure data structures
;; (maps, vectors of floats). No MemorySegments leak into user code.

(defn capture-config
  "Configure activation capture.

  layers: seq of layer indices to capture, e.g. [0 23 47]
  max-tokens: maximum number of tokens to capture (default: 256)

  Works on both CPU and Metal backends. Call before generating tokens."
  ([session layers]
   (capture-config session layers 256))
  ([session layers max-tokens]
   (n/session-capture-config session layers max-tokens)))

(defn capture-clear
  "Clear activation capture buffer and disable capture."
  [session]
  (n/session-capture-clear session))

(defn capture-info
  "Read capture buffer metadata.
  Returns a map:
    {:n-tokens N, :n-layers N, :hidden-dim N, :capacity N}
  or nil if capture is not enabled."
  [session]
  (let [^java.lang.foreign.MemorySegment out-seg (vp/alloc 16 4)]
    (when (= 1 (n/session-capture-info session out-seg))
      {:n-tokens  (int (.get out-seg java.lang.foreign.ValueLayout/JAVA_INT 0))
       :n-layers  (int (.get out-seg java.lang.foreign.ValueLayout/JAVA_INT 4))
       :hidden-dim (int (.get out-seg java.lang.foreign.ValueLayout/JAVA_INT 8))
       :capacity  (int (.get out-seg java.lang.foreign.ValueLayout/JAVA_INT 12))})))

(defn activation-get
  "Get the activation vector for a specific token index and layer index.
  Returns a vector of 2048 floats, or nil if out of bounds."
  [session token-idx layer-idx]
  (when-let [info (capture-info session)]
    (when-let [^java.lang.foreign.MemorySegment seg
               (n/activation-buffer-get (n/session-capture-buffer session)
                                        token-idx layer-idx)]
      (when (pos? (.address seg))
        (let [n (int (:hidden-dim info))
              sized (.reinterpret seg (* n 4))]
          (vec (for [i (range n)]
                 (.get sized java.lang.foreign.ValueLayout/JAVA_FLOAT (* i 4)))))))))

(defn capture-activations
  "Retrieve ALL captured activations as pure Clojure data.

  Returns a map:
    {:n-tokens    N
     :n-layers    N
     :hidden-dim  N
     :capacity    N
     :layer-indices [...]   ; the layers you configured
     :activations  [[[f1 f2 ...]    ; token 0, layer 0 (2048 floats)
                      [f1 f2 ...]    ; token 0, layer 1
                      [f1 f2 ...]]   ; token 0, layer 2
                     [[f1 f2 ...]    ; token 1, layer 0
                      ...]
                     ...]}

  Every inner vector is a 2048-element float vector representing the
  residual stream at that (token, layer) position."
  [session layer-indices]
  (when-let [info (capture-info session)]
    (let [n-toks  (:n-tokens info)
          n-lays  (:n-layers info)]
      {:n-tokens      n-toks
       :n-layers      n-lays
       :hidden-dim    (:hidden-dim info)
       :capacity      (:capacity info)
       :layer-indices (vec layer-indices)
       :activations   (vec
                        (for [t (range n-toks)]
                          (vec
                            (for [l (range n-lays)]
                              (activation-get session t l)))))})))

(defn activation-stats
  "Compute basic statistics on an activation vector.
  Returns {:mean M, :std-dev S, :min MN, :max MX, :norm L2}."
  [activations]
  (when (seq activations)
    (let [n      (count activations)
          sum    (reduce + activations)
          mean   (/ sum n)
          sqsum  (reduce + (map #(Math/pow (- % mean) 2) activations))
          std    (Math/sqrt (/ sqsum n))
          mn     (reduce min activations)
          mx     (reduce max activations)
          l2     (Math/sqrt (reduce + (map #(* % %) activations)))]
      {:mean     (float mean)
       :std-dev  (float std)
       :min      (float mn)
       :max      (float mx)
       :norm     (float l2)})))

(defn activation-norm
  "L2 norm of an activation vector."
  [activations]
  (Math/sqrt (reduce + (map #(* % %) activations))))

(defn activation-cosine-similarity
  "Cosine similarity between two activation vectors.
  Returns a float in [-1, 1], or nil if either vector is empty."
  [a b]
  (when (and (seq a) (seq b))
    (let [dot (reduce + (map * a b))
          na  (activation-norm a)
          nb  (activation-norm b)]
      (when (and (> na 0) (> nb 0))
        (float (/ dot (* na nb)))))))

;; --- Conformal Certification ---
;; Statistical guarantees for reasoning trace correctness.
;; Based on conformal prediction — provides coverage guarantees without
;; distributional assumptions.

(defn token-probabilities
  "Get the probability distribution over the vocabulary after the last
  evaluated token. Returns a sorted vector of {:id token :prob probability}
  for the top-k tokens, or all tokens if k is not provided.

  The probabilities are computed by softmax over the logits."
  [session & {:keys [k vocab-size]
               :or {k 10 vocab-size 151936}}]
  (let [logits-seg (n/session-copy-logits session vocab-size)
        logits (vec (for [i (range vocab-size)]
                      (.get logits-seg java.lang.foreign.ValueLayout/JAVA_FLOAT (* i 4))))
        max-logit (reduce max logits)
        exp-logits (mapv #(Math/exp (- % max-logit)) logits)
        sum-exp (reduce + exp-logits)
        probs (mapv #(/ % sum-exp) exp-logits)
        indexed (mapv (fn [i p] {:id i :prob (float p)}) (range vocab-size) probs)
        sorted (sort-by :prob > indexed)]
    (vec (take k sorted))))

(defn conformal-calibrate
  "Calibrate a conformal predictor on a set of calibration examples.

  Each example is a map with:
    :prompt     - the input prompt
    :answer     - the expected answer string (used to compute correctness)
    :extract-fn - (optional) function to extract answer from generated text

  Returns a calibration map:
    {:threshold T            ; the (1-alpha) quantile non-conformity score
     :alpha alpha            ; the miscoverage rate
     :scores [...]           ; all non-conformity scores from calibration
     :n-calibration N}       ; number of calibration examples

  The non-conformity score for each token is: -log(prob) where prob is the
  model's probability of the generated token. Lower scores = more confident.
  The threshold is the (1-alpha) quantile of these scores."
  [engine session calibration-examples alpha
   & {:keys [n-tokens temperature system think-mode]
      :or {n-tokens 50 temperature 0.0 think-mode :max}}]
  (let [scores (vec
                 (for [example calibration-examples]
                   (let [prompt (:prompt example)
                         _ (n/session-sync session
                                           (encode-prompt engine system prompt think-mode))
                         tokens (generate-tokens engine session prompt
                                                 :n-tokens n-tokens
                                                 :temperature temperature
                                                 :system system
                                                 :think-mode think-mode)
                         ;; For each generated token, get its probability
                         token-scores (mapv
                                        (fn [token]
                                          (let [probs (token-probabilities session :k 1)
                                                prob (or (:prob (first (filter #(= (:id %) token) probs)))
                                                         1e-10)]
                                            (- (Math/log prob))))
                                        tokens)]
                     ;; Average negative log-likelihood = non-conformity
                     (if (seq token-scores)
                       (/ (reduce + token-scores) (count token-scores))
                       0.0))))
        sorted-scores (sort scores)
        n (count sorted-scores)
        idx (int (Math/ceil (* (- 1.0 alpha) n)))
        idx (min (dec n) (max 0 (dec idx)))
        threshold (nth sorted-scores idx 0.0)]
    {:threshold threshold
     :alpha alpha
     :scores scores
     :n-calibration n}))

(defn conformal-certify-prefix
  "Certify a reasoning prefix using a calibrated conformal predictor.

  calibration: result from conformal-calibrate
  prefix-tokens: sequence of token ids in the prefix
  token-probs: sequence of probabilities (one per prefix token)

  Returns:
    {:certified? boolean
     :score float          ; the prefix non-conformity score
     :threshold float      ; the calibrated threshold
     :coverage float       ; 1 - alpha (the guaranteed coverage)
     :confidence float}    ; 1 - (score / threshold), clamped to [0,1]

  A prefix is certified if its non-conformity score is <= the threshold.
  This guarantees that (1-alpha) fraction of correct prefixes will be
  certified (marginal coverage guarantee)."
  [calibration prefix-tokens token-probs]
  (let [threshold (:threshold calibration)
        alpha (:alpha calibration)
        scores (mapv (fn [prob]
                       (- (Math/log (max prob 1e-10))))
                     token-probs)
        score (if (seq scores)
                (/ (reduce + scores) (count scores))
                0.0)
        certified? (<= score threshold)
        confidence (if (> threshold 0)
                     (max 0.0 (min 1.0 (- 1.0 (/ score threshold))))
                     0.0)]
    {:certified? certified?
     :score (float score)
     :threshold (float threshold)
     :coverage (float (- 1.0 alpha))
     :confidence (float confidence)}))

(defn conformal-certify-generation
  "Certify an entire generation by checking each prefix.

  engine, session: DS4 engine and session
  prompt: the input prompt
  calibration: result from conformal-calibrate

  Returns a map:
    {:text string              ; the generated text
     :tokens [...]             ; token ids
     :prefixes [{:text string  ; prefix text
                  :certified? boolean
                  :score float
                  :confidence float}
                ...]
     :fully-certified? boolean} ; true if ALL prefixes are certified

  This is useful for reasoning chains: you can stop generation early
  if the model loses confidence (prefix not certified)."
  [engine session prompt calibration
   & {:keys [n-tokens temperature system think-mode]
      :or {n-tokens 50 temperature 0.0 think-mode :max}}]
  (let [tokens (generate-tokens engine session prompt
                                :n-tokens n-tokens
                                :temperature temperature
                                :system system
                                :think-mode think-mode)
        text (str/join (map #(n/token-text engine %) tokens))
        ;; For each prefix, get the token probabilities
        prefixes (mapv
                   (fn [idx]
                     (let [prefix-toks (subvec tokens 0 (inc idx))
                           prefix-text (str/join (map #(n/token-text engine %) prefix-toks))
                           ;; Re-evaluate to get probabilities at this point
                           token (nth tokens idx)
                           probs (token-probabilities session :k 5)
                           token-prob (or (:prob (first (filter #(= (:id %) token) probs)))
                                          1e-10)
                           cert (conformal-certify-prefix calibration prefix-toks [token-prob])]
                       {:text prefix-text
                        :token token
                        :token-prob (float token-prob)
                        :certified? (:certified? cert)
                        :score (:score cert)
                        :confidence (:confidence cert)}))
                   (range (count tokens)))]
    {:text text
     :tokens tokens
     :prefixes prefixes
     :fully-certified? (every? :certified? prefixes)}))

;; --- Expert Routing Log ---

(defn expert-log-enable!
  "Enable expert routing logging on a session. max-entries defaults to 256."
  [session & {:keys [max-entries] :or {max-entries 256}}]
  (n/expert-log-enable session max-entries))

(defn expert-log-disable!
  "Disable expert routing logging on a session."
  [session]
  (n/expert-log-disable session))

(defn expert-log-entries
  "Read all expert log entries from a session. Returns a vector of maps."
  [session]
  (let [out-seg (vp/alloc 8 4)
        enabled? (n/expert-log-info session out-seg)]
    (if (zero? enabled?)
      []
      (let [n-entries (.get ^MemorySegment out-seg (ValueLayout/JAVA_INT) 0)]
        (if-not (pos? n-entries)
          []
          (let [log-ptr (n/expert-log-ptr session)]
            (loop [i 0
                   result []]
              (if (>= i n-entries)
                result
                (let [entry-ptr (n/expert-log-get log-ptr i)]
                  (recur (inc i)
                         (if entry-ptr
                           (conj result
                                 {:layer-idx (.get ^MemorySegment entry-ptr (ValueLayout/JAVA_INT) 0)
                                  :token-idx (.get ^MemorySegment entry-ptr (ValueLayout/JAVA_INT) 4)
                                  :selected (vec (for [j (range 8)]
                                                   (.get ^MemorySegment entry-ptr (ValueLayout/JAVA_INT) (+ 8 (* j 4)))))
                                  :weights (vec (for [j (range 8)]
                                                  (.get ^MemorySegment entry-ptr (ValueLayout/JAVA_FLOAT) (+ 40 (* j 4)))))})
                           result)))))))))))

(defn expert-log-summary
  "Summarize expert log entries: count unique experts per layer, average weights, etc.
  Returns a map with :by-layer and :by-token summaries."
  [entries]
  (let [by-layer (group-by :layer-idx entries)
        by-token (group-by :token-idx entries)]
    {:total-entries (count entries)
     :n-layers (count by-layer)
     :n-tokens (count by-token)
     :by-layer (into (sorted-map)
                     (map (fn [[layer es]]
                            [layer {:count (count es)
                                    :unique-experts (into #{} (mapcat :selected) es)
                                    :avg-top-weight (when (seq es)
                                                      (/ (reduce + (map #(first (:weights %)) es))
                                                         (count es)))}])
                          by-layer))
     :by-token (into (sorted-map)
                     (map (fn [[token es]]
                            [token {:count (count es)
                                    :layers-involved (into #{} (map :layer-idx) es)}])
                          by-token))}))

;; --- Speculative Decoding ---

(defn eval-speculative-argmax
  "Evaluate a token using speculative decoding (MTP-based draft + verify).

  Uses the model's own MTP head to draft future tokens, then verifies them
  with the full model. Can accept 1-3 tokens per call vs 1 for regular eval.

  Returns a vector of accepted token ids. On CPU, falls back to [first-token]."
  [session first-token max-tokens eos-token]
  (n/eval-speculative-argmax session first-token max-tokens eos-token))

(defn generate-speculative
  "Generate text using speculative decoding for 2-3x speedup.

  Uses ds4_session_eval_speculative_argmax internally. The MTP drafter
  proposes tokens which are verified in batches by the full model.

  NOTE: Currently only supports greedy (temperature=0) decoding.
        Sampling with speculative decode requires per-token sampling
        which defeats the batching advantage.

  Options:
    :n-tokens     - max tokens to generate (default 50)
    :system       - system prompt (default nil)
    :think-mode   - :none, :normal, or :max (default :none)

  Returns the generated text string."
  [engine session prompt & {:keys [n-tokens system think-mode]
                            :or {n-tokens 50 think-mode :none}}]
  (let [tokens (encode-prompt engine system prompt think-mode)]
    (try
      (n/session-sync session tokens)
      (let [eos (n/token-eos engine)
            result (loop [generated []]
                     (if (>= (count generated) n-tokens)
                       generated
                       (let [first-tok (n/session-argmax session)
                             accepted (n/eval-speculative-argmax
                                        session first-tok
                                        (- n-tokens (count generated))
                                        eos)]
                         (if (seq accepted)
                           (let [new-gen (into generated accepted)
                                 ;; Truncate at EOS if present
                                 eos-idx (.indexOf ^java.util.List new-gen eos)]
                             (if (>= eos-idx 0)
                               (subvec new-gen 0 (inc eos-idx))
                               (recur new-gen)))
                           generated))))]
        (str/join (map #(n/token-text engine %) result)))
      (finally
        (n/tokens-free tokens)))))

;; --- Utils ---

(defn eos-token
  "Get the EOS token id."
  [engine]
  (n/token-eos engine))

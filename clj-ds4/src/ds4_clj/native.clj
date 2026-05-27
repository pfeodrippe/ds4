(ns ds4-clj.native
  "Low-level Panama FFI bindings to libds4.dylib using vybe.panama."
  (:require
   [clojure.string :as str]
   [vybe.panama :as vp])
  (:import
   [java.lang.foreign Arena MemorySegment ValueLayout SymbolLookup FunctionDescriptor]
   [java.lang.invoke MethodHandles]))

(set! *warn-on-reflection* true)

;; --- Library loading ---

(def ^:private ds4-dir
  (or (System/getenv "DS4_DIR")
      (str (System/getProperty "user.home") "/dev/ds4")))

(def ^:private ds4-lib-path
  (str ds4-dir "/libds4.dylib"))

(def ^:private ds4-lookup
  (delay
    (SymbolLookup/libraryLookup ds4-lib-path (vp/default-arena))))

(defn- lookup-symbol
  [name]
  (-> @ds4-lookup (.find name) .get))

;; --- C struct components ---

;; NOTE: These must exactly match the C struct layouts in ds4.h.
;; Use `check_struct.c` to verify offsets and sizes.

(def DS4SteeringVector
  (vp/make-component 'DS4SteeringVector
    [[:file :string]
     [:attn_scale :float]
     [:ffn_scale :float]
     [:layer_scales_file :string]]))

(def DS4EngineOptions
  (vp/make-component 'DS4EngineOptions
    [[:model_path :string]
     [:mtp_path :string]
     [:backend :int]
     [:n_threads :int]
     [:mtp_draft_tokens :int]
     [:mtp_margin :float]
     [:directional_steering_file :string]
     [:directional_steering_attn :float]
     [:directional_steering_ffn :float]
     [:steering_vectors [:vec {:size 8} DS4SteeringVector]]
     [:n_steering_vectors :int]
     [:power_percent :int]
     [:warm_weights :byte]
     [:quality :byte]
     ;; C compiler pads to 8-byte alignment (256 total)
     [:_padding [:padding {:size 6}]]]))

(def DS4Tokens
  (vp/make-component 'DS4Tokens
    [[:v [:* :int]]
     [:len :int]
     [:cap :int]]))

(def DS4TokenScore
  (vp/make-component 'DS4TokenScore
    [[:id :int]
     [:logit :float]
     [:logprob :float]]))

;; --- Enums ---

(def backend
  {:metal 0 :cuda 1 :cpu 2})

(def think-mode
  {:none 0 :normal 1 :max 2})

;; --- Function descriptors ---

(defn- fd
  "Helper for function descriptors."
  [ret args]
  (vp/fn-descriptor
    {:type :function
     :args (mapv (fn [[sym schema]] {:symbol (name sym) :schema schema}) args)
     :ret {:schema ret}}))

;; --- C function bindings ---

(def ^:private c-engine-open
  (vp/c-fn (lookup-symbol "ds4_engine_open")
           (fd :int [[:out [:* [:* :void]]]
                     [:opts [:* :void]]])))

(def ^:private c-engine-close
  (vp/c-fn (lookup-symbol "ds4_engine_close")
           (fd :void [[:engine [:* :void]]])))

(def ^:private c-session-create
  (vp/c-fn (lookup-symbol "ds4_session_create")
           (fd :int [[:out [:* [:* :void]]]
                     [:engine [:* :void]]
                     [:ctx_size :int]])))

(def ^:private c-session-free
  (vp/c-fn (lookup-symbol "ds4_session_free")
           (fd :void [[:session [:* :void]]])))

(def ^:private c-session-sync
  (vp/c-fn (lookup-symbol "ds4_session_sync")
           (fd :int [[:session [:* :void]]
                     [:tokens [:* :void]]
                     [:err [:* :byte]]
                     [:errlen :long]])))

(def ^:private c-session-eval
  (vp/c-fn (lookup-symbol "ds4_session_eval")
           (fd :int [[:session [:* :void]]
                     [:token :int]
                     [:err [:* :byte]]
                     [:errlen :long]])))

(def ^:private c-session-sample
  (vp/c-fn (lookup-symbol "ds4_session_sample")
           (fd :int [[:session [:* :void]]
                     [:temperature :float]
                     [:top_k :int]
                     [:top_p :float]
                     [:min_p :float]
                     [:rng [:* :long]]])))

(def ^:private c-session-argmax
  (vp/c-fn (lookup-symbol "ds4_session_argmax")
           (fd :int [[:session [:* :void]]])))

(def ^:private c-session-copy-logits
  (vp/c-fn (lookup-symbol "ds4_session_copy_logits")
           (fd :int [[:session [:* :void]]
                     [:out [:* :float]]
                     [:cap :int]])))

(def ^:private c-encode-chat-prompt
  (vp/c-fn (lookup-symbol "ds4_encode_chat_prompt")
           (fd :void [[:engine [:* :void]]
                      [:system :string]
                      [:user :string]
                      [:think_mode :int]
                      [:out [:* :void]]])))

(def ^:private c-tokens-free
  (vp/c-fn (lookup-symbol "ds4_tokens_free")
           (fd :void [[:tokens [:* :void]]])))

(def ^:private c-token-text
  (vp/c-fn (lookup-symbol "ds4_token_text")
           (fd :string [[:engine [:* :void]]
                        [:token_id :int]
                        [:len [:* :long]]])))

(def ^:private c-token-eos
  (vp/c-fn (lookup-symbol "ds4_token_eos")
           (fd :int [[:engine [:* :void]]])))

(def ^:private c-set-logit-bias
  (vp/c-fn (lookup-symbol "ds4_session_set_logit_bias")
           (fd :void [[:session [:* :void]]
                      [:token_id :int]
                      [:bias :float]])))

(def ^:private c-clear-logit-bias
  (vp/c-fn (lookup-symbol "ds4_session_clear_logit_bias")
           (fd :void [[:session [:* :void]]])))

(def ^:private c-set-cfg
  (vp/c-fn (lookup-symbol "ds4_session_set_cfg")
           (fd :int [[:session [:* :void]]
                     [:scale :float]
                     [:uncond_tokens [:* :void]]])))

(def ^:private c-clear-cfg
  (vp/c-fn (lookup-symbol "ds4_session_clear_cfg")
           (fd :void [[:session [:* :void]]])))

(def ^:private c-layer-logprobs
  (vp/c-fn (lookup-symbol "ds4_session_layer_logprobs")
           (fd :int [[:session [:* :void]]
                     [:layer :int]
                     [:out [:* :void]]
                     [:k :int]])))

(def ^:private c-load-sae
  (vp/c-fn (lookup-symbol "ds4_engine_load_sae")
           (fd :int [[:engine [:* :void]]
                     [:path :string]])))

(def ^:private c-sae-steering-set
  (vp/c-fn (lookup-symbol "ds4_session_sae_steering_set")
           (fd :int [[:session [:* :void]]
                     [:feature_id :int]
                     [:scale :float]])))

(def ^:private c-sae-steering-clear
  (vp/c-fn (lookup-symbol "ds4_session_sae_steering_clear")
           (fd :void [[:session [:* :void]]])))

(def ^:private c-sae-steering-multi
  (vp/c-fn (lookup-symbol "ds4_session_sae_steering_multi")
           (fd :int [[:session [:* :void]]
                     [:n_features :int]
                     [:feature_ids [:* :int]]
                     [:scales [:* :float]]])))

;; --- Helpers ---

(defn- alloc-ptr
  "Allocate memory for a pointer (8 bytes on 64-bit)."
  []
  (vp/alloc (ValueLayout/ADDRESS)))

(defn- read-ptr
  "Read a pointer from a pointer-sized memory segment."
  [^MemorySegment seg]
  (.get seg (ValueLayout/ADDRESS) 0))

(defn- alloc-err-buf
  "Allocate an error buffer."
  ([] (alloc-err-buf 256))
  ([size]
   (let [seg (vp/alloc size 1)]
     (.fill seg 0)
     seg)))

(defn- err-string
  "Read a C string from an error buffer."
  [^MemorySegment err-buf]
  (vp/->string err-buf))

;; --- Public API ---

(defn engine-open
  "Open a DS4 engine. Returns the engine pointer or throws on error."
  [^MemorySegment opts-seg]
  (let [out (alloc-ptr)
        rc (c-engine-open out opts-seg)]
    (if (zero? rc)
      (read-ptr out)
      (throw (ex-info "ds4_engine_open failed" {:rc rc})))))

(defn engine-close
  "Close a DS4 engine."
  [^MemorySegment engine]
  (c-engine-close engine))

(defn session-create
  "Create a DS4 session. Returns the session pointer or throws on error."
  [^MemorySegment engine ctx-size]
  (let [out (alloc-ptr)
        rc (c-session-create out engine ctx-size)]
    (if (zero? rc)
      (read-ptr out)
      (throw (ex-info "ds4_session_create failed" {:rc rc})))))

(defn session-free
  "Free a DS4 session."
  [^MemorySegment session]
  (c-session-free session))

(defn session-sync
  "Sync a prompt into the session. Returns nil on success, throws on error."
  [^MemorySegment session ^MemorySegment tokens-seg]
  (let [err-buf (alloc-err-buf)
        rc (c-session-sync session tokens-seg err-buf 256)]
    (when (not= 0 rc)
      (throw (ex-info (str "ds4_session_sync failed: " (err-string err-buf))
                      {:rc rc})))))

(defn session-eval
  "Evaluate a token. Returns nil on success, throws on error."
  [^MemorySegment session token]
  (let [err-buf (alloc-err-buf)
        rc (c-session-eval session token err-buf 256)]
    (when (not= 0 rc)
      (throw (ex-info (str "ds4_session_eval failed: " (err-string err-buf))
                      {:rc rc})))))

(defn session-sample
  "Sample the next token. Returns token id."
  ([^MemorySegment session]
   (session-sample session 1.0 0 1.0 0.05))
  ([^MemorySegment session temperature top-k top-p min-p]
   ;; C library requires non-null rng when temperature > 0; allocate a dummy.
   (let [rng (vp/alloc (ValueLayout/JAVA_LONG))]
     (.set rng (ValueLayout/JAVA_LONG) 0 (long (rand-int 1000000)))
     (c-session-sample session temperature top-k top-p min-p rng))))

(defn session-argmax
  "Get the argmax token."
  [^MemorySegment session]
  (c-session-argmax session))

(defn session-copy-logits
  "Copy logits into a float array. Returns the array."
  [^MemorySegment session vocab-size]
  (let [seg (vp/alloc (* vocab-size 4) 4)
        n (c-session-copy-logits session seg vocab-size)]
    (when (not= n vocab-size)
      (throw (ex-info "ds4_session_copy_logits returned wrong size" {:expected vocab-size :actual n})))
    seg))

(defn encode-chat-prompt
  "Encode a chat prompt into a DS4Tokens struct."
  [^MemorySegment engine system user think-mode-int]
  (let [tokens-seg (vp/alloc (.layout DS4Tokens))]
    (.fill tokens-seg (byte 0))
    (c-encode-chat-prompt engine system user think-mode-int tokens-seg)
    tokens-seg))

(defn tokens-free
  "Free the internal array of a DS4Tokens struct."
  [^MemorySegment tokens-seg]
  (c-tokens-free tokens-seg))

(defn token-text
  "Get the text for a token id. NOTE: leaks memory (C malloc)."
  [^MemorySegment engine token-id]
  (let [len-ptr (vp/alloc (ValueLayout/JAVA_LONG))]
    (-> (c-token-text engine token-id len-ptr)
        vp/->string)))

(defn token-eos
  "Get the EOS token id."
  [^MemorySegment engine]
  (c-token-eos engine))

(defn set-logit-bias
  "Set logit bias for a token."
  [^MemorySegment session token-id bias]
  (c-set-logit-bias session token-id bias))

(defn clear-logit-bias
  "Clear all logit biases."
  [^MemorySegment session]
  (c-clear-logit-bias session))

(defn set-cfg
  "Enable CFG with given scale and unconditional tokens."
  [^MemorySegment session scale uncond-tokens-seg]
  (let [rc (c-set-cfg session scale uncond-tokens-seg)]
    (when (not= 0 rc)
      (throw (ex-info "ds4_session_set_cfg failed" {:rc rc})))))

(defn clear-cfg
  "Disable CFG."
  [^MemorySegment session]
  (c-clear-cfg session))

(defn layer-logprobs
  "Get top-k logprobs for a given layer. Returns a sequence of maps."
  [^MemorySegment session layer k]
  (let [out-size (* k (.byteSize (.layout DS4TokenScore)))
        out-seg (vp/alloc out-size (.byteAlignment (.layout DS4TokenScore)))
        n (c-layer-logprobs session layer out-seg k)]
    (mapv (fn [i]
            (let [score-seg (.asSlice out-seg (* i (.byteSize (.layout DS4TokenScore))))
                  pmap (vp/p->map score-seg DS4TokenScore)]
              {:id (:id pmap)
               :logit (:logit pmap)
               :logprob (:logprob pmap)}))
          (range n))))

(defn load-sae
  "Load a SAE decoder file."
  [^MemorySegment engine path]
  (let [rc (c-load-sae engine path)]
    (when (not= 0 rc)
      (throw (ex-info "ds4_engine_load_sae failed" {:rc rc :path path})))))

(defn sae-steering-set
  "Enable SAE steering."
  [^MemorySegment session feature-id scale]
  (let [rc (c-sae-steering-set session feature-id scale)]
    (when (not= 0 rc)
      (throw (ex-info "ds4_session_sae_steering_set failed" {:rc rc})))))

(defn sae-steering-clear
  "Disable SAE steering."
  [^MemorySegment session]
  (c-sae-steering-clear session))

(defn sae-steering-multi
  "Enable multi-feature SAE steering on a session.

  features: sequence of [feature-id scale] pairs, e.g. [[5 10.0] [3 -5.0]]
  Max 8 features."
  [^MemorySegment session features]
  (let [n (count features)]
    (when (> n 8)
      (throw (ex-info "Too many SAE features (max 8)" {:n n})))
    (let [^MemorySegment ids-seg (vp/alloc (* n 4) 4)
          ^MemorySegment scales-seg (vp/alloc (* n 4) 4)]
      (doseq [[i [fid scale]] (map-indexed vector features)]
        (.set ids-seg (ValueLayout/JAVA_INT) (* i 4) (int fid))
        (.set scales-seg (ValueLayout/JAVA_FLOAT) (* i 4) (float scale)))
      (let [rc (c-sae-steering-multi session n ids-seg scales-seg)]
        (when (not= 0 rc)
          (throw (ex-info "ds4_session_sae_steering_multi failed" {:rc rc})))))))

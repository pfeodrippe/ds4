(ns scratch
  "Executable REPL workflows for the Qwen3-Coder DS4 engine.

  Load with `(load-file \"scratch.clj\")`, then call a named workflow,
  `run-smoke-workflows!`, or `run-all-workflows!`."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [ds4-clj.core :as ds4]
            [ds4-clj.tools :as tools])
  (:import [java.io FileInputStream FileOutputStream]
           [java.net URI]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files StandardCopyOption]
           [java.util.zip ZipInputStream]))

(def repo-root
  (let [cwd (io/file (System/getProperty "user.dir"))]
    (.getAbsolutePath
     (cond
       (.exists (io/file cwd "qwen3-coder.gguf")) cwd
       (.exists (io/file cwd ".." "qwen3-coder.gguf")) (io/file cwd "..")
       :else (io/file (System/getProperty "user.home") "dev" "ds4")))))

(def model-path (str repo-root "/qwen3-coder.gguf"))
(def safety-steering-path (str repo-root "/dir-steering/out/safety_refusal_v3.f32"))
(def hedging-steering-path (str repo-root "/dir-steering/out/hedging_suppress_v2.f32"))
(def sarcastic-steering-path (str repo-root "/dir-steering/out/sarcastic_v1.f32"))

(defn- keyword-args [m]
  (mapcat (fn [[k v]] [k v]) m))

(defn- engine-opts [opts]
  (merge {:model-path model-path :backend :metal} opts))

(defn- call-with-engine
  ([f] (call-with-engine {} f))
  ([opts f]
   (let [engine (apply ds4/open-engine (keyword-args (engine-opts opts)))]
     (try
       (f engine)
       (finally
         (ds4/close-engine engine))))))

(defn- call-with-multi-steer-engine [vectors f]
  (let [engine (ds4/open-engine-multi-steer
                :model-path model-path
                :backend :metal
                :vectors vectors)]
    (try
      (f engine)
      (finally
        (ds4/close-engine engine)))))

(defn- call-with-session [engine ctx-size f]
  (let [session (ds4/create-session engine ctx-size)]
    (try
      (f session)
      (finally
        (ds4/free-session session)))))

(defn- call-with-engine-session
  ([f] (call-with-engine-session {} 512 f))
  ([opts ctx-size f]
   (call-with-engine opts
     (fn [engine]
       (call-with-session engine ctx-size
         (fn [session]
           (f engine session)))))))

(defn- generate! [engine session prompt opts]
  (apply ds4/generate engine session prompt (keyword-args opts)))

(defn- generate-once!
  ([prompt opts] (generate-once! {} prompt opts))
  ([opts prompt gen-opts]
   (call-with-engine-session opts 512
     (fn [engine session]
       (generate! engine session prompt gen-opts)))))

(defn- prompt-top-probabilities! [engine prompt]
  (call-with-session engine 512
    (fn [session]
      (let [tokens (ds4/encode-prompt engine nil prompt :none)]
        (try
          (ds4/session-sync session tokens)
          (ds4/token-probabilities session :k 20)
          (finally
            (ds4/tokens-free tokens)))))))

(defn- top-dims [activations n]
  (->> activations
       (map-indexed vector)
       (sort-by (fn [[_ value]] (Math/abs (double value))) >)
       (take n)
       (mapv (fn [[dim value]] {:dim dim :value (float value)}))))

(defn workflow-basic-generation!
  "Basic greedy, sampled, and thinking-mode generation."
  []
  (call-with-engine-session
   (fn [engine session]
     {:greedy (generate! engine session "The capital of France is"
                         {:n-tokens 10 :temperature 0.0})
      :sampled (generate! engine session "The capital of France is"
                          {:n-tokens 10 :temperature 0.8})
      :thinking (generate! engine session "What is 2+2?"
                           {:n-tokens 30 :think-mode :max})})))

(defn workflow-interactive-iteration!
  "Compare prompt/system/temperature changes using one engine."
  []
  (call-with-engine-session
   (fn [engine session]
     {:brief (generate! engine session "Explain quantum computing"
                        {:n-tokens 30 :temperature 0.0})
      :thinking (generate! engine session "Explain quantum computing"
                           {:n-tokens 60 :think-mode :max :temperature 0.0})
      :professor (generate! engine session "Explain quantum computing"
                            {:n-tokens 60
                             :system "You are a professor. Be thorough."
                             :think-mode :max
                             :temperature 0.0})
      :creative (generate! engine session "Write a haiku about AI"
                           {:n-tokens 30 :temperature 1.2})})))

(defn workflow-activation-capture!
  "Capture and summarize hidden states from early, middle, and late layers."
  []
  (call-with-engine-session
   (fn [engine session]
     (let [layers [0 24 47]]
       (ds4/capture-config session layers 16)
       (try
         (let [text (generate! engine session "The sky is"
                               {:n-tokens 5 :temperature 0.0})
               info (ds4/capture-info session)
               early (ds4/activation-get session 0 0)
               late (ds4/activation-get session 0 2)
               all (ds4/capture-activations session layers)]
           {:text text
            :info info
            :early-stats (ds4/activation-stats early)
            :late-stats (ds4/activation-stats late)
            :early-late-cosine (ds4/activation-cosine-similarity early late)
            :shape [(:n-tokens all) (:n-layers all) (:hidden-dim all)]})
         (finally
           (ds4/capture-clear session)))))))

(defn workflow-directional-steering!
  "Compare baseline and sarcastic directional-steering behavior."
  []
  {:baseline (generate-once! "Thank you!" {:n-tokens 20 :temperature 0.0})
   :sarcastic (generate-once! {:steering-file sarcastic-steering-path
                               :steering-ffn -2.0}
                              "Thank you!"
                              {:n-tokens 20 :temperature 0.0})})

(defn workflow-logit-lens!
  "Inspect the model's top predictions at multiple layers."
  []
  (call-with-engine-session
   (fn [engine session]
     (ds4/logit-lens engine session
                     "The capital of France is"
                     [0 6 12 18 24 30 36 42 47]
                     :k 3))))

(defn workflow-cfg!
  "Compare baseline and classifier-free-guided generation."
  []
  (call-with-engine-session
   (fn [engine session]
     (let [baseline (generate! engine session "I feel"
                               {:n-tokens 15 :temperature 0.0})
           _ (ds4/enable-cfg engine session 1.5 "I feel")
           cfg-1-5 (generate! engine session "I feel"
                             {:n-tokens 15 :temperature 0.0})
           _ (ds4/enable-cfg engine session 2.0 "I feel")
           cfg-2 (generate! engine session "I feel"
                           {:n-tokens 15 :temperature 0.0})]
       (ds4/disable-cfg session)
       {:baseline baseline :cfg-1.5 cfg-1-5 :cfg-2.0 cfg-2}))))

(defn workflow-logit-bias!
  "Exercise token banning, token boosting, and bias clearing."
  []
  (call-with-engine-session
   (fn [engine session]
     (ds4/set-logit-bias session (ds4/eos-token engine) -100.0)
     (let [without-eos (generate! engine session "Once upon a time"
                                  {:n-tokens 20 :temperature 0.0})
           _ (ds4/clear-logit-bias session)
           magic-token (first (ds4/tokenize engine "magic"))
           _ (ds4/set-logit-bias session magic-token 20.0)
           boosted (generate! engine session "The wizard cast a"
                              {:n-tokens 10 :temperature 0.0})]
       (ds4/clear-logit-bias session)
       {:magic-token magic-token
        :without-eos without-eos
        :boosted boosted}))))

(defn write-test-sae! [path]
  (let [n-features 100
        d-model 2048
        layer 24
        rng (java.util.Random. 42)]
    (with-open [out (FileOutputStream. path)]
      (let [buf (doto (ByteBuffer/allocate 12)
                  (.order ByteOrder/LITTLE_ENDIAN)
                  (.putInt n-features)
                  (.putInt d-model)
                  (.putInt layer))]
        (.write out (.array buf) 0 12))
      (doseq [_ (range n-features)]
        (let [raw (double-array (repeatedly d-model #(.nextGaussian rng)))
              norm (Math/sqrt (reduce + (map #(* % %) raw)))
              buf (doto (ByteBuffer/allocate (* d-model 4))
                    (.order ByteOrder/LITTLE_ENDIAN))]
          (doseq [value raw]
            (.putFloat buf (float (/ value norm))))
          (.write out (.array buf) 0 (* d-model 4)))))
    path))

(defn workflow-sae-steering!
  "Load a synthetic SAE and compare baseline/single/multi-feature steering."
  []
  (let [sae-path (write-test-sae! "/tmp/ds4_scratch_sae.bin")]
    (call-with-engine-session
     (fn [engine session]
       (ds4/load-sae engine sae-path)
       (let [baseline (generate! engine session "The future of AI is"
                                 {:n-tokens 20 :temperature 0.0})
             _ (ds4/sae-steer session 42 30.0)
             single (generate! engine session "The future of AI is"
                               {:n-tokens 20 :temperature 0.0})
             _ (ds4/sae-steer-multi session [[42 30.0] [7 -20.0] [15 10.0]])
             multi (generate! engine session "The future of AI is"
                              {:n-tokens 20 :temperature 0.0})]
         (ds4/sae-unsteer session)
         {:baseline baseline :single single :multi multi})))))

(defn workflow-token-by-token!
  "Generate token IDs and their decoded pieces."
  []
  (call-with-engine-session
   (fn [engine session]
     (mapv (fn [token] {:id token :text (ds4/token-text engine token)})
           (ds4/generate-tokens engine session "To be or not to be"
                                :n-tokens 10
                                :temperature 0.0)))))

(defn test-config!
  "Run one generation configuration and return its output."
  [label & {:keys [engine-opts gen-opts]
            :or {engine-opts {} gen-opts {}}}]
  {:label label
   :output (generate-once! engine-opts
                           "Write a one-sentence story."
                           (merge {:n-tokens 30} gen-opts))})

(defn workflow-config-comparison!
  "Compare baseline, high-temperature, thinking, and steered configurations."
  []
  [(test-config! :baseline :gen-opts {:temperature 0.0})
   (test-config! :high-temperature :gen-opts {:temperature 1.1})
   (test-config! :think-max :gen-opts {:think-mode :max :temperature 0.0})
   (test-config! :sarcastic
                 :engine-opts {:steering-file sarcastic-steering-path
                               :steering-ffn -2.0}
                 :gen-opts {:temperature 0.0})])

(defn workflow-capture-analysis!
  "Capture activations and report the strongest dimensions by layer."
  []
  (call-with-engine-session
   (fn [engine session]
     (let [layers [0 12 24 36 47]]
       (ds4/capture-config session layers 32)
       (try
         (let [text (generate! engine session
                               "The theory of relativity states that"
                               {:n-tokens 10 :temperature 0.0})
               data (ds4/capture-activations session layers)]
           {:text text
            :top-dimensions
            (into {}
                  (map-indexed
                   (fn [idx layer]
                     [layer (top-dims (get-in data [:activations 0 idx]) 5)])
                   layers))})
         (finally
           (ds4/capture-clear session)))))))

(defn workflow-tool-augmented-calculator!
  "Ask the model to use the Python calculator tool."
  []
  (call-with-engine-session {} 4096
   (fn [engine session]
     (let [response (tools/generate-with-tools
                     engine session
                     "Calculate the factorial of 20. Use the Python tool."
                     :max-tool-rounds 2
                     :n-tokens 100
                     :temperature 0.0)
           tool-calls (vec (or (tools/extract-tool-calls response) []))
           expected "2432902008176640000"
           result-seen? (str/includes? response expected)]
       (when-not (and (seq tool-calls) result-seen?)
         (throw (ex-info "Calculator tool did not execute to the expected result"
                         {:tool-calls tool-calls
                          :expected expected
                          :response response})))
       {:response response
        :tool-calls tool-calls
        :result-seen? result-seen?}))))

(defn workflow-conformal-certification!
  "Exercise token probabilities and conformal prefix/generation certification."
  []
  (call-with-engine-session
   (fn [engine session]
     (ds4/generate-tokens engine session "2+2=" :n-tokens 5 :temperature 0.0)
     (let [calibration {:threshold 2.0
                        :alpha 0.1
                        :scores [0.5 1.0 1.5 2.0 2.5]
                        :n-calibration 5}]
       {:top-probabilities (ds4/token-probabilities session :k 10)
        :high-confidence (ds4/conformal-certify-prefix calibration [42] [0.9])
        :low-confidence (ds4/conformal-certify-prefix calibration [42] [0.01])
        :generation (ds4/conformal-certify-generation
                     engine session "The sky is" calibration
                     :n-tokens 5 :temperature 0.0 :think-mode :none)}))))

(defn workflow-expert-routing!
  "Capture MoE routing, suppress selected experts, and compare output."
  []
  (call-with-engine
   (fn [engine]
     (call-with-session engine 512
       (fn [session]
         (ds4/expert-log-enable! session :max-entries 128)
         (try
           (let [baseline (generate! engine session "2+2="
                                     {:n-tokens 5 :temperature 0.0})
                 entries (ds4/expert-log-entries session)
                 top-experts (vec (take 4 (:selected (first entries))))]
             (call-with-session engine 512
               (fn [suppressed-session]
                 (doseq [expert top-experts]
                   (ds4/suppress-expert! suppressed-session expert))
                 (let [suppressed (generate! engine suppressed-session "2+2="
                                             {:n-tokens 5 :temperature 0.0})]
                   (ds4/unsuppress-all-experts! suppressed-session)
                   {:baseline baseline
                    :suppressed suppressed
                    :top-experts top-experts
                    :summary (ds4/expert-log-summary entries)}))))
           (finally
             (ds4/expert-log-disable! session))))))))

(defn workflow-speculative-decoding!
  "Compare self-speculative and regular greedy decoding."
  []
  (call-with-engine-session
   (fn [engine session]
     {:speculative (ds4/generate-speculative engine session
                                             "The capital of France is"
                                             :n-tokens 15)
      :regular (generate! engine session "The capital of France is"
                          {:n-tokens 15 :temperature 0.0})})))

(defn write-test-adapter! [path]
  (let [rank 1
        d-model 2048
        q-output-dim 4096
        alpha 64.0
        a (float-array (for [d (range d-model)] (if (< d 64) 10.0 0.0)))
        b (float-array (for [d (range q-output-dim)] (if (< d 64) 50.0 0.0)))]
    (with-open [out (FileOutputStream. path)]
      (let [buf (doto (ByteBuffer/allocate 256)
                  (.order ByteOrder/LITTLE_ENDIAN)
                  (.put (byte-array (map byte (concat (vec "DS4LORA") [0]))) 0 8)
                  (.putInt 1)
                  (.putInt rank)
                  (.putFloat alpha)
                  (.putInt 1)
                  (.put (byte-array 232) 0 232))]
        (.write out (.array buf) 0 256))
      (let [buf (doto (ByteBuffer/allocate 16)
                  (.order ByteOrder/LITTLE_ENDIAN)
                  (.putInt 0)
                  (.putInt 0)
                  (.putInt d-model)
                  (.putInt q-output-dim))]
        (.write out (.array buf) 0 16))
      (doseq [values [a b]]
        (let [buf (doto (ByteBuffer/allocate (* (count values) 4))
                    (.order ByteOrder/LITTLE_ENDIAN))]
          (doseq [value values]
            (.putFloat buf value))
          (.write out (.array buf) 0 (* (count values) 4)))))
    path))

(def lora-training-examples
  [{:prompt "Write a Clojure function to reverse a list"
    :response "(defn reverse-list [xs] (into () xs))"}
   {:prompt "Write a Clojure function for factorial"
    :response "(defn factorial [n] (if (<= n 1) 1 (* n (factorial (dec n)))))"}
   {:prompt "Write a Clojure function to filter even numbers"
    :response "(defn even-numbers [xs] (filter even? xs))"}])

(defn workflow-lora-pipeline!
  "Exercise adapter load/infer/unload. Pass `:train? true` for real MLX training."
  [& {:keys [train?] :or {train? false}}]
  (let [adapter-path
        (if train?
          (let [data-path (ds4/lora-prepare-data!
                           "/tmp/ds4_scratch_lora/train.jsonl"
                           lora-training-examples)]
            (ds4/lora-train! :data data-path
                             :output-dir "/tmp/ds4_scratch_lora"
                             :iters 50))
          (write-test-adapter! "/tmp/ds4_scratch_test_adapter.bin"))]
    (call-with-engine
     (fn [engine]
       (let [baseline-probs (prompt-top-probabilities!
                             engine "Write a Clojure function to reverse a list")
             baseline (call-with-session engine 512
                        #(generate! engine %
                                    "Write a Clojure function to reverse a list"
                                    {:n-tokens 30 :temperature 0.0}))
             _ (ds4/lora-load! engine adapter-path)
             adapted-probs (prompt-top-probabilities!
                            engine "Write a Clojure function to reverse a list")
             adapted (call-with-session engine 512
                       #(generate! engine %
                                   "Write a Clojure function to reverse a list"
                                   {:n-tokens 30 :temperature 0.0}))
             enabled? (ds4/lora-enabled? engine)
             logits-changed? (not= baseline-probs adapted-probs)]
         (when-not logits-changed?
           (throw (ex-info "Loaded LoRA adapter did not change prompt logits"
                           {:adapter-path adapter-path})))
         (ds4/lora-free! engine)
         {:adapter-path adapter-path
          :trained? train?
          :enabled-during-inference? enabled?
          :disabled-after-free? (not (ds4/lora-enabled? engine))
          :logits-changed? logits-changed?
          :output-changed? (not= baseline adapted)
          :baseline baseline
          :adapted adapted})))))

(defn workflow-named-models!
  "Load baseline and adapted models under separate REPL IDs and infer with both."
  []
  (let [adapter-path (write-test-adapter! "/tmp/ds4_scratch_named_adapter.bin")
        prompt "Write a Clojure function to reverse a list"]
    (try
      (ds4/load-model! :baseline
                       :model-path model-path
                       :backend :metal
                       :ctx-size 512)
      (ds4/load-model! :adapted
                       :model-path model-path
                       :backend :metal
                       :ctx-size 512
                       :adapter-path adapter-path)
      (let [baseline (ds4/generate-loaded :baseline prompt
                                          :n-tokens 20
                                          :temperature 0.0)
            adapted (ds4/generate-loaded :adapted prompt
                                         :n-tokens 20
                                         :temperature 0.0)
            adapted-probs (prompt-top-probabilities!
                           (:engine (ds4/loaded-model :adapted)) prompt)
            baseline-again (ds4/generate-loaded :baseline prompt
                                                :n-tokens 20
                                                :temperature 0.0)
            baseline-probs (prompt-top-probabilities!
                            (:engine (ds4/loaded-model :baseline)) prompt)
            logits-differ? (not= baseline-probs adapted-probs)
            baseline-restored? (= baseline baseline-again)]
        (when-not (and logits-differ? baseline-restored?)
          (throw (ex-info "Named adapter switching did not preserve model state"
                          {:logits-differ? logits-differ?
                           :baseline-restored? baseline-restored?})))
        {:loaded-models (set (ds4/loaded-model-ids))
         :outputs-differ? (not= baseline adapted)
         :logits-differ? logits-differ?
         :baseline-restored? baseline-restored?
         :baseline baseline
         :adapted adapted
         :baseline-again baseline-again})
      (finally
        (ds4/unload-all-models!)))))

(def sms-spam-zip-url
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")

(def sms-code-label {"ham" "ALPHA17" "spam" "OMEGA42"})

(defn download-sms-spam! []
  (let [dir (io/file "/tmp/ds4_sms_spam")
        zip-file (io/file dir "smsspamcollection.zip")
        data-file (io/file dir "SMSSpamCollection")]
    (.mkdirs dir)
    (when-not (.exists zip-file)
      (with-open [in (-> sms-spam-zip-url URI/create .toURL .openStream)]
        (Files/copy in (.toPath zip-file)
                    (into-array java.nio.file.CopyOption
                                [StandardCopyOption/REPLACE_EXISTING]))))
    (when-not (.exists data-file)
      (with-open [zis (ZipInputStream. (FileInputStream. zip-file))]
        (loop [entry (.getNextEntry zis)]
          (when entry
            (if (= "SMSSpamCollection" (.getName entry))
              (Files/copy zis (.toPath data-file)
                          (into-array java.nio.file.CopyOption
                                      [StandardCopyOption/REPLACE_EXISTING]))
              (recur (.getNextEntry zis)))))))
    (.getAbsolutePath data-file)))

(defn read-sms-spam []
  (->> (slurp (download-sms-spam!) :encoding "ISO-8859-1")
       str/split-lines
       (keep (fn [line]
               (let [[label text] (str/split line #"\t" 2)]
                 (when (and label text) {:label label :text text}))))
       vec))

(defn sms-prompt [text]
  (str "Classify this SMS as spam or ham. "
       "Reply with exactly one word: spam or ham.\n\nSMS: " text))

(defn sms-coded-prompt [text]
  (str "Classify this SMS using project-specific labels. "
       "Reply with exactly one label: ALPHA17 or OMEGA42.\n\nSMS: " text))

(defn- sms-prediction [coded? output]
  (let [s (if coded? (str/upper-case output) (str/lower-case output))]
    (cond
      (and coded? (str/includes? s "ALPHA17")) "ham"
      (and coded? (str/includes? s "OMEGA42")) "spam"
      (re-find #"\bspam\b" s) "spam"
      (re-find #"\bham\b" s) "ham"
      :else "unknown")))

(defn sms-balanced-split
  [& {:keys [train-per-label eval-per-label]
      :or {train-per-label 400 eval-per-label 25}}]
  (let [by-label (group-by :label (read-sms-spam))
        split-label (fn [label]
                      {:train (take train-per-label (get by-label label))
                       :eval (take eval-per-label
                                   (drop train-per-label (get by-label label)))})
        ham (split-label "ham")
        spam (split-label "spam")]
    {:train (vec (interleave (:train ham) (:train spam)))
     :eval (vec (interleave (:eval ham) (:eval spam)))}))

(defn prepare-sms-lora-data!
  [& {:keys [coded? train-per-label eval-per-label]
      :or {coded? true train-per-label 400 eval-per-label 25}}]
  (let [{:keys [train eval]} (sms-balanced-split
                              :train-per-label train-per-label
                              :eval-per-label eval-per-label)
        prompt-fn (if coded? sms-coded-prompt sms-prompt)
        response-fn (if coded? #(get sms-code-label (:label %)) :label)
        dir (if coded? "/tmp/ds4_sms_coded_lora" "/tmp/ds4_sms_lora")
        examples (map (fn [{:keys [text] :as row}]
                        {:prompt (prompt-fn text)
                         :response (response-fn row)})
                      train)]
    {:train-file (ds4/lora-prepare-data! (str dir "/train.jsonl") examples)
     :output-dir dir
     :coded? coded?
     :eval eval}))

(defn eval-sms-model! [model-id eval-examples coded?]
  (let [prompt-fn (if coded? sms-coded-prompt sms-prompt)
        rows (mapv
              (fn [{:keys [label text]}]
                (let [output (ds4/generate-loaded
                              model-id (prompt-fn text)
                              :n-tokens (if coded? 8 3)
                              :temperature 0.0)
                      prediction (sms-prediction coded? output)]
                  {:expert label
                   :prediction prediction
                   :correct? (= label prediction)
                   :output output
                   :text text}))
              eval-examples)
        correct (count (filter :correct? rows))]
    {:accuracy (/ correct (double (count rows)))
     :correct correct
     :total (count rows)
     :rows rows}))

(defn workflow-sms-finetune!
  "Train and compare a real UCI SMS adapter. This is intentionally opt-in."
  [& {:keys [coded? iters train-per-label eval-per-label]
      :or {coded? true iters 360 train-per-label 400 eval-per-label 10}}]
  (let [{:keys [train-file output-dir eval]}
        (prepare-sms-lora-data! :coded? coded?
                                :train-per-label train-per-label
                                :eval-per-label eval-per-label)
        adapter (ds4/lora-train! :data train-file
                                 :output-dir output-dir
                                 :rank 8
                                 :alpha 128
                                 :iters iters
                                 :lr 2e-4
                                 :batch-size 1
                                 :max-seq-len 192
                                 :num-layers 16)]
    (try
      (ds4/load-model! :sms :model-path model-path :backend :metal :ctx-size 512)
      (let [baseline (eval-sms-model! :sms eval coded?)
            _ (ds4/load-model-adapter! :sms adapter)
            fine-tuned (eval-sms-model! :sms eval coded?)]
        {:adapter adapter
         :baseline (select-keys baseline [:accuracy :correct :total])
         :fine-tuned (select-keys fine-tuned [:accuracy :correct :total])
         :changed-rows (count
                        (filter true?
                                (map #(not= (:prediction %1) (:prediction %2))
                                     (:rows baseline)
                                     (:rows fine-tuned))))})
      (finally
        (ds4/unload-all-models!)))))

(defn workflow-qwen-velocity!
  "Measure the multi-steered Metal decode path used by the target CLI command."
  [& {:keys [n-tokens] :or {n-tokens 220}}]
  (call-with-multi-steer-engine
   [{:file safety-steering-path :ffn 2.0}
    {:file hedging-steering-path :ffn 2.0}]
   (fn [engine]
     (call-with-session engine 4096
       (fn [session]
         (let [prompt-tokens (ds4/encode-prompt engine nil
                                                "How to be happy with myself?"
                                                :none)]
           (try
             (ds4/session-sync session prompt-tokens)
             (let [eos (ds4/eos-token engine)
                   t0 (System/nanoTime)
                   generated
                   (loop [i 0 out []]
                     (if (>= i n-tokens)
                       out
                       (let [token (ds4/session-sample session 0.7 0 1.0 0.05)]
                         (if (or (= token eos) (< token 0))
                           out
                           (do
                             (ds4/session-eval session token)
                             (recur (inc i) (conj out token)))))))
                   seconds (/ (- (System/nanoTime) t0) 1.0e9)]
               {:tokens (count generated)
                :seconds seconds
                :tok-per-s (/ (count generated) seconds)
                :preview (apply str
                                (map #(ds4/token-text engine %)
                                     (take 60 generated)))})
             (finally
               (ds4/tokens-free prompt-tokens)))))))))

(def workflow-registry
  [{:id :qwen-velocity :group :standard :run workflow-qwen-velocity!}
   {:id :basic-generation :group :standard :run workflow-basic-generation!}
   {:id :interactive-iteration :group :standard :run workflow-interactive-iteration!}
   {:id :activation-capture :group :standard :run workflow-activation-capture!}
   {:id :directional-steering :group :standard :run workflow-directional-steering!}
   {:id :logit-lens :group :standard :run workflow-logit-lens!}
   {:id :cfg :group :standard :run workflow-cfg!}
   {:id :logit-bias :group :standard :run workflow-logit-bias!}
   {:id :sae-steering :group :standard :run workflow-sae-steering!}
   {:id :token-by-token :group :standard :run workflow-token-by-token!}
   {:id :config-comparison :group :standard :run workflow-config-comparison!}
   {:id :capture-analysis :group :standard :run workflow-capture-analysis!}
   {:id :tool-augmented-calculator :group :standard :run workflow-tool-augmented-calculator!}
   {:id :conformal-certification :group :standard :run workflow-conformal-certification!}
   {:id :speculative-decoding :group :standard :run workflow-speculative-decoding!}
   {:id :lora-pipeline :group :standard :run workflow-lora-pipeline!}
   {:id :named-models :group :slow :run workflow-named-models!}
   {:id :expert-routing :group :slow :run workflow-expert-routing!}
   {:id :sms-real-finetune :group :training :run workflow-sms-finetune!}])

(defn run-workflow!
  "Run one workflow descriptor and return timing, status, and result/error."
  [{:keys [id group run]}]
  (println (format "[scratch] running %-28s (%s)" (name id) (name group)))
  (let [t0 (System/nanoTime)]
    (try
      (let [result (run)
            seconds (/ (- (System/nanoTime) t0) 1.0e9)]
        (println (format "[scratch] passed  %-28s %.2fs" (name id) seconds))
        {:id id :group group :status :passed :seconds seconds :result result})
      (catch Throwable error
        (let [seconds (/ (- (System/nanoTime) t0) 1.0e9)]
          (println (format "[scratch] FAILED  %-28s %.2fs: %s"
                           (name id) seconds (.getMessage error)))
          {:id id
           :group group
           :status :failed
           :seconds seconds
           :error (Throwable->map error)})))))

(defn run-workflows!
  "Run selected workflow groups and return a summary.

  Groups are `:standard`, `:slow`, and `:training`."
  [& {:keys [groups]
      :or {groups #{:standard}}}]
  (let [selected (filter #(contains? groups (:group %)) workflow-registry)
        results (mapv run-workflow! selected)]
    {:passed (count (filter #(= :passed (:status %)) results))
     :failed (count (filter #(= :failed (:status %)) results))
     :results results}))

(defn run-smoke-workflows!
  "Run every bounded, non-training workflow."
  []
  (run-workflows! :groups #{:standard}))

(defn run-all-workflows!
  "Run standard and slow workflows. Pass `:include-training? true` to also
  download the UCI dataset and run the real MLX fine-tune."
  [& {:keys [include-training?]
      :or {include-training? false}}]
  (run-workflows! :groups (cond-> #{:standard :slow}
                            include-training? (conj :training))))

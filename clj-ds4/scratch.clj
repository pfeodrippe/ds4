(ns scratch
  (:require [clojure.string :as str]
            [ds4-clj.core :as ds4])
  (:import [java.io FileInputStream FileOutputStream]
           [java.net URI]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files StandardCopyOption]
           [java.util.zip ZipInputStream]))

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
;; WORKFLOW 17: Real Dataset Fine-Tune — UCI SMS Spam Collection
;; =============================================================================
;;
;; Dataset:
;;   UCI SMS Spam Collection, 5,574 labeled SMS messages.
;;   https://archive.ics.uci.edu/dataset/228/sms+spam+collection
;;
;; Goal:
;;   Train a LoRA adapter that makes Qwen3-Coder answer one-word SMS labels:
;;   "spam" or "ham". Then compare base model, fine-tuned adapter, and the
;;   dataset label as the expert/ground truth.

(def sms-spam-zip-url
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")

(defn download-sms-spam! []
  (let [dir (java.io.File. "/tmp/ds4_sms_spam")
        zip-file (java.io.File. dir "smsspamcollection.zip")
        data-file (java.io.File. dir "SMSSpamCollection")]
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
                 (when (and label text)
                   {:label label :text text}))))))

(defn sms-prompt [text]
  (str "Classify this SMS as spam or ham. "
       "Reply with exactly one word: spam or ham.\n\nSMS: " text))

(def sms-code-label
  {"ham" "ALPHA17"
   "spam" "OMEGA42"})

(defn sms-coded-prompt [text]
  (str "Classify this SMS using the project-specific labels. "
       "Reply with exactly one label: ALPHA17 or OMEGA42.\n\nSMS: " text))

(defn sms-example->training [{:keys [label text]}]
  {:prompt (sms-prompt text)
   :response label})

(defn sms-example->coded-training [{:keys [label text]}]
  {:prompt (sms-coded-prompt text)
   :response (get sms-code-label label)})

(defn sms-balanced-split
  "Return balanced train/eval examples from the UCI data."
  [& {:keys [train-per-label eval-per-label]
      :or {train-per-label 400 eval-per-label 25}}]
  (let [by-label (group-by :label (read-sms-spam))
        select-label (fn [label]
                       (let [xs (get by-label label)]
                         {:train (take train-per-label xs)
                          :eval (take eval-per-label (drop train-per-label xs))}))
        ham (select-label "ham")
        spam (select-label "spam")]
    {:train (interleave (:train ham) (:train spam))
     :eval (interleave (:eval ham) (:eval spam))}))

(defn prepare-sms-lora-data! []
  (let [{:keys [train eval]} (sms-balanced-split)
        train-file (ds4/lora-prepare-data! "/tmp/ds4_sms_lora/train.jsonl"
                                           (map sms-example->training train))]
    {:train-file train-file
     :eval (vec eval)}))

(defn prepare-sms-coded-lora-data! []
  (let [{:keys [train eval]} (sms-balanced-split)
        train-file (ds4/lora-prepare-data! "/tmp/ds4_sms_coded_lora/train.jsonl"
                                           (map sms-example->coded-training train))]
    {:train-file train-file
     :eval (vec eval)}))

(defn sms-prediction [model-output]
  (let [s (str/lower-case model-output)]
    (cond
      (re-find #"\bspam\b" s) "spam"
      (re-find #"\bham\b" s) "ham"
      :else "unknown")))

(defn sms-coded-prediction [model-output]
  (let [s (str/upper-case model-output)]
    (cond
      (str/includes? s "ALPHA17") "ham"
      (str/includes? s "OMEGA42") "spam"
      :else "unknown")))

(defn eval-sms-model! [model-id eval-examples]
  (let [rows (for [{:keys [label text]} eval-examples
                   :let [out (ds4/generate-loaded model-id (sms-prompt text)
                                                  :n-tokens 3
                                                  :temperature 0.0)
                         pred (sms-prediction out)]]
               {:model model-id
                :expert label
                :prediction pred
                :correct? (= label pred)
                :output out
                :text text})
        total (count rows)
        correct (count (filter :correct? rows))]
    {:model model-id
     :accuracy (if (pos? total) (/ correct (double total)) 0.0)
     :correct correct
     :total total
     :rows (vec rows)}))

(defn eval-sms-coded-model! [model-id eval-examples]
  (let [rows (for [{:keys [label text]} eval-examples
                   :let [out (ds4/generate-loaded model-id (sms-coded-prompt text)
                                                  :n-tokens 8
                                                  :temperature 0.0)
                         pred (sms-coded-prediction out)]]
               {:model model-id
                :expert label
                :prediction pred
                :correct? (= label pred)
                :output out
                :text text})
        total (count rows)
        correct (count (filter :correct? rows))]
    {:model model-id
     :accuracy (if (pos? total) (/ correct (double total)) 0.0)
     :correct correct
     :total total
     :rows (vec rows)}))

(defn compare-sms-models! [eval-examples]
  {:baseline (eval-sms-model! :baseline eval-examples)
   :fine-tuned (eval-sms-model! :fine-tuned eval-examples)})

(defn sms-balanced-eval-sample [eval-examples n-per-label]
  (let [by-label (group-by :label eval-examples)]
    (vec (concat (take n-per-label (get by-label "ham"))
                 (take n-per-label (get by-label "spam"))))))

(comment
  ;; --- Step 1: prepare a real dataset ---
  (def sms-data (prepare-sms-lora-data!))
  (:train-file sms-data)
  (count (:eval sms-data))

  ;; --- Step 2: train a new adapter from the Clojure REPL ---
  ;; MLX expects a directory with train.jsonl; ds4/lora-train! accepts either
  ;; that directory or the train.jsonl file path returned above.
  (def sms-adapter
    (ds4/lora-train! :data (:train-file sms-data)
                     :output-dir "/tmp/ds4_sms_lora"
                     :model "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
                     :rank 8
                     :alpha 128
                     :iters 240
                     :lr 2e-4
                     :batch-size 1
                     :max-seq-len 192
                     :num-layers 16))

  ;; --- Step 3: load multiple named models in the REPL ---
  ;; :baseline is the frozen Qwen3-Coder model.
  ;; :fine-tuned is the same base model with the trained DS4 adapter loaded.
  (ds4/load-model! :baseline
                   :model-path "qwen3-coder.gguf"
                   :backend :metal
                   :ctx-size 512)

  (ds4/load-model! :fine-tuned
                   :model-path "qwen3-coder.gguf"
                   :backend :metal
                   :ctx-size 512
                   :adapter-path sms-adapter)

  (ds4/loaded-model-ids)

  ;; --- Step 4: compare baseline vs fine-tuned vs expert labels ---
  (def eval-sample (sms-balanced-eval-sample (:eval sms-data) 10))
  (def sms-results (compare-sms-models! eval-sample))
  (select-keys (:baseline sms-results) [:accuracy :correct :total])
  (select-keys (:fine-tuned sms-results) [:accuracy :correct :total])

  ;; Inspect rows where the adapter changed the baseline answer.
  (->> (map vector (get-in sms-results [:baseline :rows])
            (get-in sms-results [:fine-tuned :rows]))
       (filter (fn [[b ft]] (not= (:prediction b) (:prediction ft))))
       (take 5))

  ;; --- Memory-conscious variant: one engine, adapter toggled in-place ---
  ;; This still compares base vs adapter without loading two 30B engines.
  (ds4/unload-model! :fine-tuned)
  (def before (eval-sms-model! :baseline eval-sample))
  (ds4/load-model-adapter! :baseline sms-adapter)
  (def after (eval-sms-model! :baseline eval-sample))
  (ds4/free-model-adapter! :baseline)

  (ds4/unload-all-models!)
  )

(comment
  ;; Stronger fine-tuning demonstration: same real UCI dataset, but labels are
  ;; project-specific codes. The base model cannot infer the arbitrary mapping
  ;; reliably from the prompt alone; the adapter has to learn it.
  (def coded-data (prepare-sms-coded-lora-data!))
  (def coded-adapter
    (ds4/lora-train! :data (:train-file coded-data)
                     :output-dir "/tmp/ds4_sms_coded_lora"
                     :model "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
                     :rank 8
                     :alpha 128
                     :iters 360
                     :lr 2e-4
                     :batch-size 1
                     :max-seq-len 192
                     :num-layers 16))

  (def coded-eval-sample (sms-balanced-eval-sample (:eval coded-data) 10))
  (ds4/load-model! :baseline
                   :model-path "qwen3-coder.gguf"
                   :backend :metal
                   :ctx-size 512)
  (def coded-before (eval-sms-coded-model! :baseline coded-eval-sample))
  (ds4/load-model-adapter! :baseline coded-adapter)
  (def coded-after (eval-sms-coded-model! :baseline coded-eval-sample))
  (select-keys coded-before [:accuracy :correct :total])
  (select-keys coded-after [:accuracy :correct :total])
  (ds4/unload-all-models!)
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

  ;; --- Step 5: Expert suppression (dropping) ---
  ;; You can suppress specific experts to steer model behavior.
  ;; This is useful for safety research: identify "refusal" experts
  ;; and suppress them to study model behavior.
  (def s2 (ds4/create-session engine 512))
  (ds4/expert-log-enable! s2 :max-entries 256)
  (ds4/generate engine s2 "2+2=" {:n-tokens 3 :temperature 0.0})
  (def entries2 (ds4/expert-log-entries s2))
  (def top-experts (take 4 (:selected (first entries2))))
  (println "\nTop-4 experts for first token:" top-experts)

  ;; Suppress them and regenerate
  (doseq [eid top-experts]
    (ds4/suppress-expert! s2 eid))
  (def suppressed-text (ds4/generate engine s2 "2+2="
                                      {:n-tokens 10 :temperature 0.0}))
  (println "Suppressed output:" suppressed-text)

  ;; Clear suppression
  (ds4/unsuppress-all-experts! s2)
  (ds4/free-session s2)

  ;; --- Step 6: Disable logging ---
  (ds4/expert-log-disable! session)

  (ds4/free-session session)
  (ds4/close-engine engine)
  )

;; =========================================================================
;; Workflow 15: Self-Speculative Decoding
;;
;; NOTE: On Qwen3-Coder 30B-A3B, this does NOT provide a speedup.
;; Benchmark: 0.93× regular speed (431ms vs 402ms). The MTP draft overhead
;; ≈ tokens saved. May help on larger models or longer sequences.
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

;; Helper: create a synthetic test adapter (no training needed)
(defn write-test-adapter [path]
  (let [rank 8
        d-model 2048
        q-output-dim 4096
        alpha 16.0
        n-layers 1
        A (float-array (for [r (range rank) d (range d-model)]
                         (* 0.01 (- (Math/random) 0.5))))
        B (float-array (for [d (range q-output-dim) r (range rank)]
                         (* 0.01 (- (Math/random) 0.5))))]
    (with-open [out (FileOutputStream. path)]
      (let [buf (ByteBuffer/allocate 256)]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (.put buf (byte-array (map byte (concat (vec "DS4LORA") [0]))) 0 8)
        (.putInt buf 1) (.putInt buf rank) (.putFloat buf alpha)
        (.putInt buf n-layers) (.put buf (byte-array 232) 0 232)
        (.write out (.array buf) 0 256))
      (let [buf (ByteBuffer/allocate 16)]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (.putInt buf 0) (.putInt buf 0) (.putInt buf d-model) (.putInt buf q-output-dim)
        (.write out (.array buf) 0 16))
      (let [buf (ByteBuffer/allocate (* rank d-model 4))]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (doseq [v A] (.putFloat buf v))
        (.write out (.array buf) 0 (* rank d-model 4)))
      (let [buf (ByteBuffer/allocate (* q-output-dim rank 4))]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (doseq [v B] (.putFloat buf v))
        (.write out (.array buf) 0 (* q-output-dim rank 4))))
    path))

;; WORKFLOW 16: LoRA (Low-Rank Adaptation) — Full Pipeline
;; =============================================================================
;; Fine-tune the model from the REPL. The adapter is a separate data structure
;; loaded alongside the base model — both coexist in the same engine.
;;
;; Architecture:
;;   Base model (frozen, ~17GB GGUF)
;;   + Adapter (trainable, ~10MB)
;;   = Fine-tuned behavior
;;
;; The adapter can be loaded/unloaded at runtime without restarting the engine.
;; This makes it a first-class data structure you manipulate from the REPL.
;;
;; NOTE: Training uses Python/MLX under the hood (gradient computation).
;;       The REPL orchestrates the pipeline — you never leave Clojure.
(comment
  (require '[ds4-clj.core :as ds4])

  ;; --- Step 0: Open engine (base model) ---
  (def engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal))
  (println "Base model loaded. LoRA enabled?" (ds4/lora-enabled? engine))
  ;; => false

  ;; ===================================================================
  ;; PATH A: Full REPL-native training (prepare → train → load)
  ;; ===================================================================

  ;; --- Step 1: Prepare training data (pure Clojure) ---
  (def training-data
    [{:prompt "Write a Clojure function to reverse a list"
      :response "(defn reverse-list [lst] (into () lst))"}
     {:prompt "Write a Clojure function for factorial"
      :response "(defn factorial [n] (if (<= n 1) 1 (* n (factorial (dec n)))))"}
     {:prompt "Write a Clojure function to filter even numbers"
      :response "(defn even-numbers [coll] (filter even? coll))"}
     ;; ... add more examples
     ])

  (ds4/lora-prepare-data! "/tmp/my_ft_data.jsonl" training-data)

  ;; --- Step 2: Train adapter (REPL calls Python/MLX) ---
  ;; This returns the path to the converted DS4 adapter
  (def adapter-path
    (ds4/lora-train! :data "/tmp/my_ft_data.jsonl"
                     :output-dir "/tmp/my_adapter"
                     :model "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
                     :rank 8
                     :alpha 16
                     :iters 50))
  ;; => "/tmp/my_adapter/ds4_lora.bin"

  ;; --- Step 3: Load adapter into engine (no restart needed) ---
  (ds4/lora-load! engine adapter-path)
  (println "Adapter loaded. LoRA enabled?" (ds4/lora-enabled? engine))
  ;; => true

  ;; ===================================================================
  ;; PATH B: Quick synthetic test (no training needed)
  ;; ===================================================================

  ;; Use this to verify the pipeline without waiting for training:
  ;; (write-test-adapter "/tmp/test_adapter.bin")
  ;; (ds4/lora-load! engine "/tmp/test_adapter.bin")

  ;; ===================================================================
  ;; PATH C: One-shot (train + load in one call)
  ;; ===================================================================

  ;; (ds4/lora-train-and-load! engine
  ;;   :data "/tmp/my_ft_data.jsonl"
  ;;   :output-dir "/tmp/my_adapter"
  ;;   :iters 50)

  ;; ===================================================================
  ;; INFERENCE: Generate with adapter active
  ;; ===================================================================

  ;; The adapter is automatically applied on Metal GPU during attention.
  ;; The base model weights stay frozen — only the adapter deltas are applied.
  (def session-with (ds4/create-session engine 512))
  (def text-with (ds4/generate engine session-with
                                "Write a Clojure function to reverse a list"
                                {:n-tokens 40 :temperature 0.0}))
  (println "With adapter:" text-with)
  (ds4/free-session session-with)

  ;; --- Unload adapter (back to base model) ---
  (ds4/lora-free! engine)
  (println "Adapter unloaded. LoRA enabled?" (ds4/lora-enabled? engine))
  ;; => false

  (def session-without (ds4/create-session engine 512))
  (def text-without (ds4/generate engine session-without
                                   "Write a Clojure function to reverse a list"
                                   {:n-tokens 40 :temperature 0.0}))
  (println "Without adapter:" text-without)
  (ds4/free-session session-without)

  ;; --- You can load a DIFFERENT adapter without restarting ---
  ;; (ds4/lora-load! engine "/tmp/other_adapter/ds4_lora.bin")

  (ds4/close-engine engine)
  )

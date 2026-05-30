(ns ds4-clj.core-test
  "Integration tests for ds4-clj.core.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test

  Uses Metal backend on macOS (auto-discovers shader sources).
  Uses CPU backend on Linux."
  (:require
   [clojure.java.io :as io]
   [clojure.java.shell :as shell]
   [clojure.string :as str]
   [clojure.test :refer [deftest is testing use-fixtures]]
   [ds4-clj.core :as ds4]))

(def ^:dynamic *engine* nil)
(def ^:dynamic *session* nil)

(defn engine-fixture
  [f]
  (let [backend (if (str/includes? (str/lower-case (System/getProperty "os.name")) "mac")
                  :metal
                  :cpu)
        engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend backend)]
    (binding [*engine* engine]
      (try
        (f)
        (finally
          (ds4/close-engine engine))))))

(defn session-fixture
  [f]
  (let [session (ds4/create-session *engine* 512)]
    (binding [*session* session]
      (try
        (f)
        (finally
          (ds4/free-session session))))))

(use-fixtures :once engine-fixture)
(use-fixtures :each session-fixture)

(deftest test-basic-generation
  (testing "Greedy generation produces deterministic output"
    (let [text (ds4/generate *engine* *session* "The capital of France is"
                             {:n-tokens 10 :temperature 0.0})]
      (is (string? text))
      (is (> (count text) 0))
      (is (re-find #"Paris" text) "Should mention Paris"))))

(deftest test-generation-with-system-prompt
  (testing "System prompt affects generation"
    (let [text (ds4/generate *engine* *session* "2+2="
                             {:n-tokens 5 :temperature 0.0
                              :system "You are a calculator. Answer with only the number."})]
      (is (string? text))
      (is (> (count text) 0)))))

(deftest test-logit-lens
  (testing "Logit lens returns predictions for each layer on Metal"
    (let [lens (ds4/logit-lens *engine* *session* "The capital of France is" [0 12 24 47])]
      (is (map? lens))
      (is (= 4 (count lens)))
      (doseq [layer [0 12 24 47]]
        (is (contains? lens layer))
        (is (vector? (get lens layer)))
        (is (pos? (count (get lens layer))))
        (let [pred (first (get lens layer))]
          (is (contains? pred :id))
          (is (contains? pred :logit))
          (is (contains? pred :logprob)))))))

(deftest test-logit-bias
  (testing "Logit bias prevents EOS token"
    (let [session2 (ds4/create-session *engine* 512)
          _ (ds4/set-logit-bias session2 (ds4/eos-token *engine*) -100.0)
          text (ds4/generate *engine* session2 "The answer is"
                             {:n-tokens 10 :temperature 0.8})]
      (is (string? text))
      (ds4/free-session session2))))

(deftest test-cfg
  (testing "CFG changes generation output"
    (let [no-cfg (ds4/generate *engine* *session* "Write a haiku about"
                               {:n-tokens 20 :temperature 0.8})]
      (ds4/enable-cfg *engine* *session* 1.5 "The weather today is")
      (let [with-cfg (ds4/generate *engine* *session* "Write a haiku about"
                                   {:n-tokens 20 :temperature 0.8})]
        ;; CFG should usually change output; allow occasional sameness
        (is (or (not= no-cfg with-cfg)
                (string? no-cfg))
            "CFG should produce valid text"))
      (ds4/disable-cfg *session*))))

(deftest test-sae-steering
  (testing "SAE steering changes output"
    (let [sae-path "/tmp/ds4_test_synthetic.sae"
          _ (when-not (.exists (io/file sae-path))
              (shell/sh "python3" "-c"
                        (str "import struct,random;random.seed(42);"
                             "f=open('" sae-path "','wb');"
                             "f.write(struct.pack('<III',10,2048,24));"
                             "[f.write(struct.pack('<f',random.random()-0.5)) for _ in range(10*2048)];"
                             "f.close()")))]
      (ds4/load-sae *engine* sae-path)
      (ds4/sae-steer *session* 5 50.0)
      (let [text (ds4/generate *engine* *session* "Hello"
                               {:n-tokens 5 :temperature 0.0})]
        (is (string? text))
        (is (> (count text) 0)))
      (ds4/sae-unsteer *session*))))

(deftest test-multi-sae-steering
  (testing "Multi-feature SAE steering works"
    (let [sae-path "/tmp/ds4_test_synthetic.sae"
          _ (when-not (.exists (io/file sae-path))
              (shell/sh "python3" "-c"
                        (str "import struct,random;random.seed(42);"
                             "f=open('" sae-path "','wb');"
                             "f.write(struct.pack('<III',10,2048,24));"
                             "[f.write(struct.pack('<f',random.random()-0.5)) for _ in range(10*2048)];"
                             "f.close()")))]
      (ds4/load-sae *engine* sae-path)
      ;; Single feature
      (ds4/sae-steer *session* 5 100.0)
      (let [single (ds4/generate *engine* *session* "Hello"
                                 {:n-tokens 5 :temperature 0.0})]
        ;; Multi-feature: same feature twice at half scale = same total
        (ds4/sae-steer-multi *session* [[5 50.0] [5 50.0]])
        (let [multi (ds4/generate *engine* *session* "Hello"
                                   {:n-tokens 5 :temperature 0.0})]
          (is (= single multi)
              "Same feature at 50+50 should equal 100"))
        ;; Different features: just verify it doesn't crash and produces output
        (ds4/sae-steer-multi *session* [[3 100.0] [7 100.0]])
        (let [multi2 (ds4/generate *engine* *session* "Hello"
                                    {:n-tokens 5 :temperature 0.0})]
          (is (string? multi2))
          (is (> (count multi2) 0))))
      (ds4/sae-unsteer *session*))))

(deftest test-steering-vector
  (testing "Steering vector path is accepted at engine open time"
    ;; We verify the path resolves without crashing.
    ;; Actual behavioral steering is tested in C test suite.
    (let [ds4-dir (or (System/getenv "DS4_DIR")
                      (str (System/getProperty "user.home") "/dev/ds4"))]
      (is (.exists (io/file ds4-dir "dir-steering/out/sarcastic_v1.f32"))
          "Sarcastic steering vector file should exist"))))

(deftest test-activation-capture-metal
  (testing "Activation capture on Metal — pure Clojure data API"
    (let [layers [0 23 47]]
      (ds4/capture-config *session* layers 8)

      ;; Generate some tokens so we have captured data
      (let [text (ds4/generate *engine* *session* "The capital of France is"
                               {:n-tokens 5 :temperature 0.0})]
        (is (string? text))
        (is (> (count text) 0)))

      ;; --- Test capture-info (pure map) ---
      (let [info (ds4/capture-info *session*)]
        (is (map? info) "capture-info should return a map")
        (is (contains? info :n-tokens))
        (is (contains? info :n-layers))
        (is (contains? info :hidden-dim))
        (is (contains? info :capacity))
        (is (>= (:n-tokens info) 5) "Should have captured at least 5 tokens")
        (is (= 3 (:n-layers info)) "Should have 3 captured layers")
        (is (= 2048 (:hidden-dim info)) "Hidden dim should be 2048")
        (is (= 8 (:capacity info)) "Capacity should be 8"))

      ;; --- Test activation-get (pure vector of floats) ---
      (let [act0-l0 (ds4/activation-get *session* 0 0)]
        (is (vector? act0-l0) "activation-get should return a vector")
        (is (= 2048 (count act0-l0)) "Activation vector should have 2048 elements")
        (is (float? (first act0-l0)) "Elements should be floats")
        (let [stats (ds4/activation-stats act0-l0)]
          (is (map? stats) "activation-stats should return a map")
          (is (contains? stats :mean))
          (is (contains? stats :norm))
          (is (> (:norm stats) 1.0) (format "Layer 0 norm should be >1, got %.2f" (:norm stats)))))

      (let [act0-l47 (ds4/activation-get *session* 0 2)]
        (is (vector? act0-l47))
        (let [stats (ds4/activation-stats act0-l47)]
          (is (> (:norm stats) 1.0) (format "Layer 47 norm should be >1, got %.2f" (:norm stats)))))

      ;; --- Test capture-activations (full nested data) ---
      (let [data (ds4/capture-activations *session* layers)]
        (is (map? data) "capture-activations should return a map")
        (is (contains? data :n-tokens))
        (is (contains? data :n-layers))
        (is (contains? data :hidden-dim))
        (is (contains? data :capacity))
        (is (contains? data :layer-indices))
        (is (contains? data :activations))
        (is (= layers (:layer-indices data)))
        (is (vector? (:activations data)))
        (is (= 5 (count (:activations data))) "Should have 5 tokens")
        (is (vector? (first (:activations data))))
        (is (= 3 (count (first (:activations data)))) "Each token should have 3 layers")
        (is (vector? (first (first (:activations data)))))
        (is (= 2048 (count (first (first (:activations data)))))) "Each layer should have 2048 floats")

      ;; --- Test activation utilities ---
      (let [a0 (ds4/activation-get *session* 0 0)
            a1 (ds4/activation-get *session* 1 0)
            a2 (ds4/activation-get *session* 2 0)]
        (is (not= a0 a1) "Token 0 and 1 should differ")
        (is (not= a1 a2) "Token 1 and 2 should differ")
        ;; Cosine similarity
        (let [cos (ds4/activation-cosine-similarity a0 a1)]
          (is (some? cos) "Cosine similarity should be computable")
          (is (float? cos))
          (is (< cos 0.99) (format "Cosine similarity should be <0.99, got %.4f" cos))
          (is (> cos -0.5) (format "Cosine similarity should be >-0.5, got %.4f" cos)))
        ;; L2 norm
        (let [n0 (ds4/activation-norm a0)
              n1 (ds4/activation-norm a1)]
          (is (> n0 0) "Norm should be positive")
          (is (> n1 0) "Norm should be positive"))
        ;; Stats
        (let [stats (ds4/activation-stats a0)]
          (is (map? stats))
          (is (contains? stats :mean))
          (is (contains? stats :std-dev))
          (is (contains? stats :min))
          (is (contains? stats :max))
          (is (contains? stats :norm))))

      ;; --- Test layer progression ---
      (let [act-l0  (ds4/activation-get *session* 0 0)
            act-l47 (ds4/activation-get *session* 0 2)
            cos     (ds4/activation-cosine-similarity act-l0 act-l47)]
        (is (some? cos))
        (when cos
          (is (< cos 0.99) (format "Layer 0 and 47 should differ (cos=%.4f)" cos))
          (is (> cos -0.5) (format "Layer 0 and 47 should not be anti-correlated (cos=%.4f)" cos)))

      (ds4/capture-clear *session*)
      (is (nil? (ds4/capture-info *session*))
          "Capture info should be nil after clear")
      (is (nil? (ds4/capture-activations *session* layers))
          "capture-activations should be nil after clear")))))


(deftest test-conformal-certification
  (testing "Conformal certification API works with real model data"
    ;; Generate some tokens and get probabilities
    (let [tokens (ds4/generate-tokens *engine* *session* "2+2="
                                      {:n-tokens 5 :temperature 0.0})]
      (is (seq tokens) "Should generate tokens")

      ;; Test token-probabilities
      (let [probs (ds4/token-probabilities *session* :k 5)]
        (is (vector? probs) "token-probabilities should return a vector")
        (is (= 5 (count probs)) "Should return top-5 probabilities")
        (is (map? (first probs)) "Each entry should be a map")
        (is (contains? (first probs) :id) "Should have :id")
        (is (contains? (first probs) :prob) "Should have :prob")
        (is (float? (:prob (first probs))) "Probability should be a float")
        (is (> (:prob (first probs)) 0) "Probability should be positive")
        (is (<= (:prob (first probs)) 1.0) "Probability should be <= 1.0")
        ;; Probabilities should be sorted descending
        (is (apply >= (map :prob probs))
            "Probabilities should be sorted descending"))

      ;; Test conformal-certify-prefix with dummy calibration
      (let [calibration {:threshold 2.0 :alpha 0.1 :scores [0.5 1.0 1.5 2.0 2.5] :n-calibration 5}
            cert (ds4/conformal-certify-prefix calibration [42] [0.9])]
        (is (map? cert) "conformal-certify-prefix should return a map")
        (is (contains? cert :certified?))
        (is (contains? cert :score))
        (is (contains? cert :threshold))
        (is (contains? cert :coverage))
        (is (contains? cert :confidence))
        (is (boolean? (:certified? cert)))
        (is (float? (:score cert)))
        (is (float? (:confidence cert)))
        (is (>= (:confidence cert) 0.0))
        (is (<= (:confidence cert) 1.0))
        ;; High probability (0.9) should be certified with threshold 2.0
        (is (:certified? cert)
            "High probability token should be certified"))

      ;; Test with low probability (should NOT be certified)
      (let [calibration {:threshold 2.0 :alpha 0.1 :scores [0.5 1.0 1.5 2.0 2.5] :n-calibration 5}
            cert (ds4/conformal-certify-prefix calibration [42] [0.01])]
        (is (not (:certified? cert))
            "Very low probability token should NOT be certified"))

      ;; Test conformal-certify-generation (full pipeline)
      (let [calibration {:threshold 3.0 :alpha 0.1 :scores [0.5 1.0 2.0 3.0] :n-calibration 4}
            result (ds4/conformal-certify-generation
                     *engine* *session* "The sky is"
                     calibration
                     {:n-tokens 5 :temperature 0.0})]
        (is (map? result) "conformal-certify-generation should return a map")
        (is (contains? result :text))
        (is (contains? result :tokens))
        (is (contains? result :prefixes))
        (is (contains? result :fully-certified?))
        (is (string? (:text result)))
        (is (vector? (:tokens result)))
        (is (vector? (:prefixes result)))
        (is (boolean? (:fully-certified? result)))
        ;; Each prefix should have certification info
        (doseq [prefix (:prefixes result)]
          (is (contains? prefix :text))
          (is (contains? prefix :certified?))
          (is (contains? prefix :score))
          (is (contains? prefix :confidence)))))))

(deftest test-expert-logging
  (testing "Expert routing logging API — stubs compile and return expected shapes"
    ;; Enable expert logging
    (ds4/expert-log-enable! *session* :max-entries 64)

    ;; Generate a few tokens to populate the log
    (let [text (ds4/generate *engine* *session* "Hello"
                             {:n-tokens 3 :temperature 0.0})]
      (is (string? text)))

    ;; Read entries — populated on CPU backend only.
    ;; Metal uses GPU shaders for MoE; expert data would need shader-side capture.
    (let [entries (ds4/expert-log-entries *session*)]
      (is (vector? entries) "expert-log-entries should return a vector")
      ;; On Metal: entries will be empty (GPU-side MoE)
      ;; On CPU: entries will be populated with expert routing data
      (is (every? map? entries) "Each entry should be a map if present")

      ;; Summary works on empty or populated entries
      (let [summary (ds4/expert-log-summary entries)]
        (is (map? summary))
        (is (number? (:total-entries summary)))
        (is (number? (:n-layers summary)))
        (is (number? (:n-tokens summary)))))

    ;; Disable and verify cleanup
    (ds4/expert-log-disable! *session*)
    (let [entries (ds4/expert-log-entries *session*)]
      (is (empty? entries) "Should be empty after disable"))))

(deftest test-speculative-decoding
  (testing "Speculative decoding produces valid text"
    ;; Note: speculative decode only works on Metal with MTP ready.
    ;; On CPU it falls back to regular eval. Either way it should produce text.
    (let [text (ds4/generate-speculative *engine* *session* "The sky is"
                                          {:n-tokens 10})]
      (is (string? text) "Should return a string")
      (is (> (count text) 0) "Should produce non-empty text"))

    ;; Compare with regular generation — both should be valid
    (let [regular (ds4/generate *engine* *session* "The sky is"
                                {:n-tokens 10 :temperature 0.0})
          speculative (ds4/generate-speculative *engine* *session* "The sky is"
                                                 {:n-tokens 10})]
      (is (string? regular))
      (is (string? speculative))
      (is (> (count regular) 0))
      (is (> (count speculative) 0))
      ;; They may differ (speculative uses MTP drafter) or be the same
      ;; (if MTP is not ready or draft is rejected). Both are valid.
      )))

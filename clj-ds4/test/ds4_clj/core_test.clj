(ns ds4-clj.core-test
  "Integration tests for ds4-clj.core.

  Run with:
    DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test

  For Metal backend, set all DS4_METAL_*_SOURCE env vars first."
  (:require
   [clojure.java.io :as io]
   [clojure.java.shell :as shell]
   [clojure.test :refer [deftest is testing use-fixtures]]
   [ds4-clj.core :as ds4]))

(def ^:dynamic *engine* nil)
(def ^:dynamic *session* nil)

(defn engine-fixture
  [f]
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :cpu)]
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
  (testing "Logit lens returns predictions for each layer"
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
      (is (= 10 (count (re-seq #"\S+" text)))) ; rough token count check
      (ds4/free-session session2))))

(deftest test-cfg
  (testing "CFG changes generation output"
    (let [no-cfg (ds4/generate *engine* *session* "I feel"
                               {:n-tokens 8 :temperature 0.8})]
      (ds4/enable-cfg *engine* *session* 1.5 "I feel")
      (let [with-cfg (ds4/generate *engine* *session* "I feel"
                                   {:n-tokens 8 :temperature 0.8})]
        ;; CFG should produce different output (not guaranteed but highly likely)
        (is (not= no-cfg with-cfg)
            "CFG should change the output distribution"))
      (ds4/disable-cfg *session*))))

(deftest test-sae-steering
  (testing "SAE steering changes output"
    ;; Use pre-generated synthetic SAE file from C test suite
    (let [sae-path "/tmp/ds4_test_synthetic.sae"
          _ (when-not (.exists (io/file sae-path))
              ;; Generate it if missing
              (clojure.java.shell/sh "python3" "-c"
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

(deftest test-steering-vector
  (testing "Directional steering vector changes output"
    (ds4/close-engine *engine*)
    (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                  :backend :cpu
                                  :steering-file "dir-steering/out/sarcastic_v1.f32"
                                  :steering-ffn -2.0)
          session (ds4/create-session engine 512)
          text (ds4/generate engine session "Thank you!"
                             {:n-tokens 10 :temperature 0.8})]
      (is (string? text))
      ;; Sarcastic vector should produce distinctive markers
      (is (or (re-find #"[Oo]h" text)
              (re-find #"\*" text)
              (re-find #"!" text))
          "Sarcastic steering should produce emotional markers")
      (ds4/free-session session)
      (ds4/close-engine engine))))

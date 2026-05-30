(ns ds4-clj.expert-cpu-test
  "Test expert routing logging on CPU backend.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.expert-cpu-test"
  (:require
   [clojure.string :as str]
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

(deftest test-expert-logging-cpu
  (testing "Expert routing logging captures real MoE data on CPU"
    ;; Enable expert logging
    (ds4/expert-log-enable! *session* :max-entries 256)

    ;; Generate a few tokens to populate the log
    (let [text (ds4/generate *engine* *session* "2+2="
                             {:n-tokens 5 :temperature 0.0})]
      (is (string? text))
      (is (> (count text) 0)))

    ;; Read entries — should be populated on CPU
    (let [entries (ds4/expert-log-entries *session*)]
      (is (vector? entries))
      (is (pos? (count entries)) "Should capture expert routing entries on CPU")

      ;; Each entry should have the expected shape
      (when (seq entries)
        (let [e (first entries)]
          (is (contains? e :layer-idx))
          (is (contains? e :token-idx))
          (is (contains? e :selected))
          (is (contains? e :weights))
          (is (= 8 (count (:selected e))) "Should have 8 selected experts")
          (is (= 8 (count (:weights e))) "Should have 8 weights")
          (is (every? int? (:selected e)) "Selected should be ints")
          (is (every? float? (:weights e)) "Weights should be floats")
          ;; Expert IDs should be in valid range (0-127 for Qwen3-Coder)
          (is (every? #(and (>= % 0) (< % 128)) (:selected e))
              "Expert IDs should be in range [0, 127]")
          ;; Weights should be positive (router weights after softmax + scale)
          (is (every? pos? (:weights e)) "Weights should be positive")))

      ;; Should have entries for multiple layers (MoE layers are every few layers)
      (let [layers (into #{} (map :layer-idx) entries)]
        (is (> (count layers) 1) "Should capture multiple layers"))

      ;; Should have entries for multiple tokens
      (let [tokens (into #{} (map :token-idx) entries)]
        (is (> (count tokens) 1) "Should capture multiple tokens"))

      ;; Summary should show meaningful data
      (let [summary (ds4/expert-log-summary entries)]
        (is (map? summary))
        (is (pos? (:total-entries summary)))
        (is (pos? (:n-layers summary)))
        (is (pos? (:n-tokens summary)))
        ;; By-layer should show unique experts per layer
        (is (seq (:by-layer summary)))
        ;; By-token should show layers involved per token
        (is (seq (:by-token summary)))))

    ;; Disable and verify cleanup
    (ds4/expert-log-disable! *session*)
    (let [entries (ds4/expert-log-entries *session*)]
      (is (empty? entries) "Should be empty after disable"))))

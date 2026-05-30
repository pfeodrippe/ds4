(ns ds4-clj.expert-suppress-test
  "Test expert suppression (dropping) on CPU backend.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.expert-suppress-test"
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [ds4-clj.core :as ds4]))

(def ^:dynamic *engine* nil)

(defn engine-fixture
  [f]
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :cpu)]
    (binding [*engine* engine]
      (try
        (f)
        (finally
          (ds4/close-engine engine))))))

(use-fixtures :once engine-fixture)

(deftest test-expert-suppression-changes-output
  (testing "Suppressing multiple experts changes generation output"
    (let [prompt "Write a haiku about autumn:"]

      ;; Baseline: generate with a fresh session
      (let [session1 (ds4/create-session *engine* 512)]
        (try
          (let [baseline (ds4/generate *engine* session1 prompt
                                        {:n-tokens 30 :temperature 0.0})]
            (is (string? baseline))
            (is (> (count baseline) 0))

            ;; Suppressed: fresh session with top-4 experts suppressed
            (let [session2 (ds4/create-session *engine* 512)]
              (try
                ;; Capture routing to find top experts, then suppress them
                (ds4/expert-log-enable! session2 :max-entries 256)
                (ds4/generate *engine* session2 prompt {:n-tokens 5 :temperature 0.0})
                (let [entries (ds4/expert-log-entries session2)
                      top-experts (take 4 (:selected (first entries)))]
                  (is (seq top-experts) "Should have captured routing data")

                  ;; Suppress the top 4 experts
                  (doseq [eid top-experts]
                    (ds4/suppress-expert! session2 eid))

                  ;; Verify they're suppressed
                  (doseq [eid top-experts]
                    (is (ds4/expert-suppressed? session2 eid)
                        (format "Expert %d should be suppressed" eid)))

                  ;; Generate again with suppression
                  (let [suppressed (ds4/generate *engine* session2 prompt
                                                  {:n-tokens 30 :temperature 0.0})]
                    (is (string? suppressed))
                    (is (> (count suppressed) 0))

                    ;; With 4 experts suppressed, output should differ
                    ;; (the model must route to different experts)
                    (is (not= baseline suppressed)
                        (format "Suppressing top-4 experts should change output\nBaseline: %s\nSuppressed: %s"
                                baseline suppressed)))

                  ;; Unsuppress all and verify output returns to baseline
                  (ds4/unsuppress-all-experts! session2)
                  (let [cleared (ds4/generate *engine* session2 prompt
                                               {:n-tokens 30 :temperature 0.0})]
                    (is (= baseline cleared)
                        "After clearing suppression, output should match baseline")))
                (finally
                  (ds4/free-session session2)))))
          (finally
            (ds4/free-session session1)))))))



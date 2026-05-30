(ns ds4-clj.expert-patterns-test
  "Verify that expert routing patterns differ across prompts.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.expert-patterns-test"
  (:require
   [clojure.set :as set]
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

(defn- get-expert-signature
  "Get a 'signature' of expert routing for a prompt.
  Returns a set of [layer-idx expert-id] pairs for the first token."
  [prompt]
  (let [session (ds4/create-session *engine* 512)]
    (try
      (ds4/expert-log-enable! session :max-entries 256)
      (ds4/generate *engine* session prompt {:n-tokens 5 :temperature 0.0})
      (let [entries (ds4/expert-log-entries session)
            ;; Get expert selections for layer 0 only (first MoE layer)
            ;; across all tokens — this is enough to distinguish prompts
            sig (into #{} (for [e entries
                                s (:selected e)]
                            [(:layer-idx e) s]))]
        sig)
      (finally
        (ds4/free-session session)))))

(deftest test-expert-patterns-differ
  (testing "Different prompts activate different expert routing patterns"
    (let [math-sig (get-expert-signature "2 + 2 =")
          code-sig (get-expert-signature "def fibonacci(n):")
          text-sig (get-expert-signature "The weather today is")]

      (is (seq math-sig) "Math prompt should produce routing data")
      (is (seq code-sig) "Code prompt should produce routing data")
      (is (seq text-sig) "Text prompt should produce routing data")

      ;; Math and code should differ (they're different domains)
      (let [overlap (set/intersection math-sig code-sig)
            total (set/union math-sig code-sig)]
        (is (< (count overlap) (count total))
            (format "Math and code should have different routing (overlap=%d, total=%d)"
                    (count overlap) (count total))))

      ;; Code and text should differ
      (let [overlap (set/intersection code-sig text-sig)
            total (set/union code-sig text-sig)]
        (is (< (count overlap) (count total))
            (format "Code and text should have different routing (overlap=%d, total=%d)"
                    (count overlap) (count total))))

      ;; Print diagnostics
      (println (format "\n=== Expert Routing Patterns ==="))
      (println (format "Math prompt:  %d unique [layer expert] pairs" (count math-sig)))
      (println (format "Code prompt:  %d unique [layer expert] pairs" (count code-sig)))
      (println (format "Text prompt:  %d unique [layer expert] pairs" (count text-sig)))
      (println (format "Math∩Code:    %d pairs" (count (set/intersection math-sig code-sig))))
      (println (format "Code∩Text:    %d pairs" (count (set/intersection code-sig text-sig))))
      (println (format "Math∩Text:    %d pairs" (count (set/intersection math-sig text-sig)))))))
